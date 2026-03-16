"""OpenAI 兼容适配服务模块。
负责将 OpenAI Chat Completions 兼容请求转换为当前系统的大模型调用。
当前阶段负责协议适配与真实流式输出，不负责会话持久化和内部业务状态管理。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from json import dumps, loads
from typing import Protocol
from uuid import uuid4

from app.clients.llm_client import (
    LlmChatCompletionChunk,
    LlmClient,
    LlmInputMessage,
    LlmToolCall,
    LlmToolCallChunk,
)
from app.core.exceptions import AppException
from app.core.logger import get_logger
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

LOGGER = get_logger(__name__)


class OpenAICompatibleResult(Protocol):
    """OpenAI 兼容响应构建所需的结果协议。"""

    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    tool_calls: list[object]


@dataclass(slots=True)
class _OpenAIStreamChunkBuilder:
    """OpenAI SSE 事件构造器。"""

    default_model_name: str
    response_id: str = ""
    created_at: int = 0
    resolved_model_name: str = ""
    has_emitted_role: bool = False
    has_emitted_finish: bool = False
    saw_tool_call_chunk: bool = False

    def __post_init__(self) -> None:
        self.response_id = f"chatcmpl-{uuid4().hex}"
        self.created_at = int(datetime.now(UTC).timestamp())
        self.resolved_model_name = self.default_model_name

    def consume_chunk(self, chunk: LlmChatCompletionChunk) -> list[str]:
        """将统一增量结果转换为一个或多个 OpenAI 兼容 chunk。"""

        if chunk.model_name:
            self.resolved_model_name = chunk.model_name

        payloads: list[str] = []
        if chunk.content_delta or chunk.tool_call_chunks:
            delta: dict[str, object] = {}
            if not self.has_emitted_role:
                delta["role"] = "assistant"
                self.has_emitted_role = True

            if chunk.content_delta:
                delta["content"] = chunk.content_delta

            if chunk.tool_call_chunks:
                self.saw_tool_call_chunk = True
                delta["tool_calls"] = [
                    self._build_stream_tool_call_payload(tool_call_chunk)
                    for tool_call_chunk in chunk.tool_call_chunks
                ]

            payloads.append(
                self._format_payload(
                    {
                        "id": self.response_id,
                        "object": "chat.completion.chunk",
                        "created": self.created_at,
                        "model": self.resolved_model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )

        if chunk.finish_reason:
            payloads.append(self._build_finish_payload(chunk.finish_reason))

        return payloads

    def finalize(self, finish_reason: str | None = None) -> list[str]:
        """在流结束时补齐结束事件和 [DONE]。"""

        payloads: list[str] = []
        if not self.has_emitted_finish:
            payloads.append(
                self._build_finish_payload(
                    finish_reason or ("tool_calls" if self.saw_tool_call_chunk else "stop")
                )
            )
        payloads.append("data: [DONE]\n\n")
        return payloads

    def _build_finish_payload(self, finish_reason: str) -> str:
        """构造结束 chunk。"""

        self.has_emitted_finish = True
        return self._format_payload(
            {
                "id": self.response_id,
                "object": "chat.completion.chunk",
                "created": self.created_at,
                "model": self.resolved_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )

    @staticmethod
    def _build_stream_tool_call_payload(tool_call_chunk: LlmToolCallChunk) -> dict[str, object]:
        """构造单个流式工具调用片段。"""

        function_payload: dict[str, object] = {}
        if tool_call_chunk.tool_name is not None:
            function_payload["name"] = tool_call_chunk.tool_name
        if tool_call_chunk.arguments_chunk or tool_call_chunk.tool_name is not None:
            function_payload["arguments"] = tool_call_chunk.arguments_chunk

        return OpenAICompatService._remove_none_values(
            {
                "index": tool_call_chunk.index,
                "id": tool_call_chunk.tool_call_id,
                "type": "function",
                "function": function_payload or None,
            }
        )

    @staticmethod
    def _format_payload(payload: dict[str, object]) -> str:
        """将字典负载转换为 SSE 文本格式。"""

        compact_payload = OpenAICompatService._remove_none_values(payload)
        return f"data: {dumps(compact_payload, ensure_ascii=False)}\n\n"


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

        completion_result = await self._execute_completion(request)
        return self.build_chat_completion_response(completion_result)

    async def stream_chat_completion(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """以 SSE 形式输出 OpenAI 兼容流式结果。"""

        input_messages = self.build_input_messages(request.messages)
        requested_tool_names = self.extract_requested_tool_names(request)
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
        stream_iterator = self._llm_client.stream_chat_completion(
            messages=input_messages,
            model_name=request.model,
            tools=selected_tools,
            tool_choice=tool_choice,
        )

        return self._stream_llm_chunks_as_openai_sse(
            llm_chunks=stream_iterator,
            default_model_name=request.model or self._llm_client.default_model_name,
        )

    def build_chat_completion_response(
        self,
        completion_result: OpenAICompatibleResult,
    ) -> OpenAIChatCompletionResponse:
        """将统一补全结果转换为 OpenAI 兼容响应。"""

        response_id = f"chatcmpl-{uuid4().hex}"
        created_at = int(datetime.now(UTC).timestamp())

        return OpenAIChatCompletionResponse(
            id=response_id,
            created=created_at,
            model=str(completion_result.model_name),
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=OpenAIChatCompletionAssistantMessage(
                        content=completion_result.content or None,
                        tool_calls=(
                            self._build_openai_tool_calls(list(completion_result.tool_calls))
                            or None
                        ),
                    ),
                    finish_reason=str(completion_result.finish_reason),
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=int(completion_result.prompt_tokens),
                completion_tokens=int(completion_result.completion_tokens),
                total_tokens=int(completion_result.total_tokens),
            ),
        )

    def create_stream_chunk_builder(self, *, default_model_name: str) -> _OpenAIStreamChunkBuilder:
        """创建流式 SSE 事件构造器。"""

        return _OpenAIStreamChunkBuilder(default_model_name=default_model_name)

    def build_stream_error_payload(self, exception: AppException) -> str:
        """构造流式错误事件，避免客户端因连接中断直接报 aborted。"""

        return _OpenAIStreamChunkBuilder._format_payload(
            {
                "error": {
                    "message": exception.message,
                    "type": "stream_error",
                    "code": exception.error_code,
                    "details": exception.details or None,
                }
            }
        )

    async def _stream_llm_chunks_as_openai_sse(
        self,
        *,
        llm_chunks: AsyncIterator[LlmChatCompletionChunk],
        default_model_name: str,
    ) -> AsyncIterator[str]:
        """将统一流式增量实时转换为 OpenAI 兼容 SSE。"""

        chunk_builder = self.create_stream_chunk_builder(default_model_name=default_model_name)
        has_emitted_payload = False

        try:
            async for llm_chunk in llm_chunks:
                for payload in chunk_builder.consume_chunk(llm_chunk):
                    has_emitted_payload = True
                    yield payload

            for payload in chunk_builder.finalize():
                has_emitted_payload = True
                yield payload
        except AppException as exception:
            if not has_emitted_payload:
                raise

            yield self.build_stream_error_payload(exception)
            yield "data: [DONE]\n\n"
        except Exception as exception:
            if not has_emitted_payload:
                raise

            LOGGER.exception("OpenAI 兼容流式输出过程中发生未处理异常。", exc_info=exception)
            yield self.build_stream_error_payload(
                AppException(
                    "流式输出过程中发生内部异常。",
                    error_code="stream_error",
                )
            )
            yield "data: [DONE]\n\n"

    async def _execute_completion(self, request: OpenAIChatCompletionRequest):
        """执行一次 OpenAI 兼容补全调用。"""

        input_messages = self.build_input_messages(request.messages)
        requested_tool_names = self.extract_requested_tool_names(request)
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

    def build_input_messages(self, messages: list[OpenAIChatMessage]) -> list[LlmInputMessage]:
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

    def extract_requested_tool_names(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> list[str] | None:
        """提取请求中允许使用的工具名称。"""

        if request.tools is None:
            return None

        requested_tool_names = [tool_definition.function.name for tool_definition in request.tools]
        self._tool_registry.ensure_supported(requested_tool_names)
        return requested_tool_names

    def extract_latest_user_message(self, messages: list[OpenAIChatMessage]) -> str:
        """提取最后一条非空 user 消息文本。"""

        for message in reversed(messages):
            if message.role == "user":
                normalized_content = self._normalize_message_content(message)
                if normalized_content:
                    return normalized_content

        raise AppException(
            "OpenAI 兼容请求至少需要包含一条非空 user 消息。",
            error_code="invalid_request",
        )

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
    def _remove_none_values(payload: object) -> object:
        """递归移除负载中的空值字段。"""

        if isinstance(payload, dict):
            return {
                key: OpenAICompatService._remove_none_values(value)
                for key, value in payload.items()
                if value is not None
            }
        if isinstance(payload, list):
            return [OpenAICompatService._remove_none_values(item) for item in payload]
        return payload
