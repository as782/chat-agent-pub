"""OpenAI 兼容适配服务模块。
负责将 OpenAI Chat Completions 兼容请求转换为当前系统的模型调用。
当前阶段负责协议适配与流式输出，不负责会话持久化和内部业务状态管理。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from json import dumps, loads
from uuid import uuid4

from app.clients.llm_client import LlmClient, LlmInputMessage, LlmToolCall
from app.core.exceptions import AppException
from app.schemas.openai_compat import (
    OpenAIChatCompletionAssistantMessage,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChatMessage,
)
from app.tools.registry import ToolRegistry


class OpenAICompatService:
    """OpenAI 兼容接口适配服务。"""

    def __init__(
        self,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._llm_client = llm_client or LlmClient()
        self._tool_registry = tool_registry or ToolRegistry()

    async def create_chat_completion(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> OpenAIChatCompletionResponse:
        """处理 OpenAI 兼容聊天请求。"""

        response_id = f"chatcmpl-{uuid4().hex}"
        created_at = int(datetime.now(UTC).timestamp())
        completion_result = await self._execute_completion(request)

        return OpenAIChatCompletionResponse(
            id=response_id,
            created=created_at,
            model=completion_result.model_name,
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=OpenAIChatCompletionAssistantMessage(
                        content=completion_result.content or None,
                        tool_calls=self._build_openai_tool_calls(completion_result.tool_calls)
                        or None,
                    ),
                    finish_reason=completion_result.finish_reason,
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=completion_result.prompt_tokens,
                completion_tokens=completion_result.completion_tokens,
                total_tokens=completion_result.total_tokens,
            ),
        )

    async def stream_chat_completion(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """以 SSE 形式输出 OpenAI 兼容流式结果。"""

        response_id = f"chatcmpl-{uuid4().hex}"
        created_at = int(datetime.now(UTC).timestamp())
        completion_result = await self._execute_completion(request)

        if completion_result.tool_calls:
            for tool_call_index, tool_call in enumerate(completion_result.tool_calls):
                chunk_payload = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": completion_result.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant" if tool_call_index == 0 else None,
                                "tool_calls": [
                                    {
                                        "index": tool_call_index,
                                        "id": tool_call.tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.tool_name,
                                            "arguments": dumps(
                                                tool_call.arguments,
                                                ensure_ascii=False,
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield self._format_sse_payload(chunk_payload)
        else:
            answer_chunks = self._split_text_for_stream(completion_result.content)
            if not answer_chunks:
                answer_chunks = [""]

            for chunk_index, answer_chunk in enumerate(answer_chunks):
                chunk_payload = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": completion_result.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant" if chunk_index == 0 else None,
                                "content": answer_chunk,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield self._format_sse_payload(chunk_payload)

        finish_payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": completion_result.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": completion_result.finish_reason,
                }
            ],
        }
        yield self._format_sse_payload(finish_payload)
        yield "data: [DONE]\n\n"

    async def _execute_completion(self, request: OpenAIChatCompletionRequest):
        """执行一次 OpenAI 兼容补全调用。"""

        input_messages = self._build_input_messages(request.messages)
        requested_tool_names = self._extract_requested_tool_names(request)
        if requested_tool_names is None and request.tool_choice is not None:
            raise AppException(
                "未传入 tools 时不能指定 tool_choice。",
                error_code="invalid_request",
            )
        selected_tools = (
            self._tool_registry.get_tools(requested_tool_names)
            if requested_tool_names is not None
            else None
        )
        tool_choice = self._tool_registry.normalize_tool_choice(request.tool_choice)

        return await self._llm_client.create_chat_completion(
            messages=input_messages,
            model_name=request.model,
            tools=selected_tools,
            tool_choice=tool_choice,
        )

    def _build_input_messages(self, messages: list[OpenAIChatMessage]) -> list[LlmInputMessage]:
        """将 OpenAI 消息转换为统一输入消息。"""

        input_messages: list[LlmInputMessage] = []
        for message in messages:
            normalized_content = self._normalize_message_content(message)
            input_messages.append(
                LlmInputMessage(
                    role=message.role,
                    content=normalized_content,
                    name=message.name,
                    tool_call_id=message.tool_call_id,
                    tool_calls=self._parse_input_tool_calls(message),
                )
            )

        return input_messages

    def _extract_requested_tool_names(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> list[str] | None:
        """提取请求中允许使用的工具名称。"""

        if request.tools is None:
            return None

        requested_tool_names = [tool_definition.function.name for tool_definition in request.tools]
        self._tool_registry.ensure_supported(requested_tool_names)
        return requested_tool_names

    def _parse_input_tool_calls(self, message: OpenAIChatMessage) -> list[LlmToolCall]:
        """解析输入 assistant 消息中的工具调用。"""

        parsed_tool_calls: list[LlmToolCall] = []
        for tool_call in message.tool_calls or []:
            try:
                arguments = loads(tool_call.function.arguments or "{}")
            except ValueError as exception:
                raise AppException(
                    "工具调用参数不是合法 JSON。",
                    error_code="invalid_request",
                    details={"tool_call_id": tool_call.id},
                ) from exception

            if not isinstance(arguments, dict):
                raise AppException(
                    "工具调用参数必须是 JSON 对象。",
                    error_code="invalid_request",
                    details={"tool_call_id": tool_call.id},
                )

            parsed_tool_calls.append(
                LlmToolCall(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=arguments,
                )
            )

        return parsed_tool_calls

    def _build_openai_tool_calls(
        self,
        tool_calls: list[LlmToolCall],
    ) -> list[OpenAIChatCompletionToolCall]:
        """将统一工具调用转换为 OpenAI 兼容输出。"""

        return [
            OpenAIChatCompletionToolCall(
                id=tool_call.tool_call_id,
                function=OpenAIChatCompletionToolCallFunction(
                    name=tool_call.tool_name,
                    arguments=dumps(tool_call.arguments, ensure_ascii=False),
                ),
            )
            for tool_call in tool_calls
        ]

    @staticmethod
    def _normalize_message_content(message: OpenAIChatMessage) -> str:
        """将消息内容统一转换为纯文本。"""

        content = message.content
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = [
                content_part.text.strip() for content_part in content if content_part.text.strip()
            ]
            return "\n".join(text_parts).strip()

        return ""

    @staticmethod
    def _split_text_for_stream(content: str, chunk_size: int = 20) -> list[str]:
        """按固定片段切分文本，用于最小流式输出。"""

        if not content:
            return []

        return [content[index : index + chunk_size] for index in range(0, len(content), chunk_size)]

    @staticmethod
    def _format_sse_payload(payload: dict[str, object]) -> str:
        """将字典负载转换为 SSE 文本格式。"""

        compact_payload = OpenAICompatService._remove_none_values(payload)
        return f"data: {dumps(compact_payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _remove_none_values(payload: object) -> object:
        """递归移除 SSE 负载中的空值字段。"""

        if isinstance(payload, dict):
            return {
                key: OpenAICompatService._remove_none_values(value)
                for key, value in payload.items()
                if value is not None
            }
        if isinstance(payload, list):
            return [OpenAICompatService._remove_none_values(item) for item in payload]
        return payload
