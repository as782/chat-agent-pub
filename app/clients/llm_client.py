"""LLM 客户端模块。
负责统一封装对外部大模型的调用，避免 service 层直接依赖第三方 SDK。
当前阶段支持普通补全、真实流式输出与工具调用，不负责多模型路由和复杂重试策略。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from json import dumps, loads
from time import perf_counter
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from app.agent.prompts import BASE_SINGLE_TURN_SYSTEM_PROMPT
from app.core.config import get_settings
from app.core.exceptions import AppException, ConfigurationException, UpstreamServiceException
from app.core.logger import get_logger

LOGGER = get_logger(__name__)

LlmBindableTool = BaseTool | dict[str, Any]


@dataclass(slots=True)
class LlmToolCall:
    """大模型返回的工具调用。"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class LlmToolCallChunk:
    """流式工具调用增量。"""

    index: int
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_chunk: str = ""


@dataclass(slots=True)
class LlmInputMessage:
    """统一的输入消息模型。"""

    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[LlmToolCall] = field(default_factory=list)


@dataclass(slots=True)
class LlmChatCompletionChunk:
    """统一的流式增量结果。"""

    content_delta: str = ""
    model_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    tool_call_chunks: list[LlmToolCallChunk] = field(default_factory=list)


@dataclass(slots=True)
class LlmChatCompletionResult:
    """大模型对话结果。"""

    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    tool_calls: list[LlmToolCall] = field(default_factory=list)


@dataclass(slots=True)
class _ToolCallAccumulator:
    """工具调用累加器。"""

    index: int
    tool_call_id: str = ""
    tool_name: str = ""
    argument_chunks: list[str] = field(default_factory=list)


class LlmChatCompletionAccumulator:
    """流式补全结果累加器。"""

    def __init__(self, *, requested_model_name: str | None, default_model_name: str) -> None:
        self._requested_model_name = requested_model_name
        self._default_model_name = default_model_name
        # 保存逐块到达的文本片段，避免 service 层重复实现拼接逻辑。
        self._content_chunks: list[str] = []
        # 保存流式工具调用分片，便于在最终阶段还原成完整 JSON 参数。
        self._tool_call_accumulators: dict[int, _ToolCallAccumulator] = {}
        self._resolved_model_name: str | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._finish_reason: str | None = None

    def append_chunk(self, chunk: LlmChatCompletionChunk) -> None:
        """累加一段流式输出。"""

        if chunk.content_delta:
            self._content_chunks.append(chunk.content_delta)
        if chunk.model_name:
            self._resolved_model_name = chunk.model_name
        if chunk.prompt_tokens is not None:
            self._prompt_tokens = chunk.prompt_tokens
        if chunk.completion_tokens is not None:
            self._completion_tokens = chunk.completion_tokens
        if chunk.total_tokens is not None:
            self._total_tokens = chunk.total_tokens
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason

        for tool_call_chunk in chunk.tool_call_chunks:
            tool_call_accumulator = self._tool_call_accumulators.setdefault(
                tool_call_chunk.index,
                _ToolCallAccumulator(index=tool_call_chunk.index),
            )
            if tool_call_chunk.tool_call_id:
                tool_call_accumulator.tool_call_id = tool_call_chunk.tool_call_id
            if tool_call_chunk.tool_name:
                tool_call_accumulator.tool_name = tool_call_chunk.tool_name
            if tool_call_chunk.arguments_chunk:
                tool_call_accumulator.argument_chunks.append(tool_call_chunk.arguments_chunk)

    def build_result(self) -> LlmChatCompletionResult:
        """构造完整补全结果。"""

        tool_calls = self._build_tool_calls()
        finish_reason = self._finish_reason or ("tool_calls" if tool_calls else "stop")
        total_tokens = self._total_tokens or (self._prompt_tokens + self._completion_tokens)
        resolved_model_name = (
            self._resolved_model_name or self._requested_model_name or self._default_model_name
        )

        return LlmChatCompletionResult(
            content="".join(self._content_chunks).strip(),
            model_name=resolved_model_name,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def _build_tool_calls(self) -> list[LlmToolCall]:
        """将流式工具调用片段还原为完整工具调用列表。"""

        tool_calls: list[LlmToolCall] = []
        for tool_call_index in sorted(self._tool_call_accumulators):
            tool_call_accumulator = self._tool_call_accumulators[tool_call_index]
            arguments_text = "".join(tool_call_accumulator.argument_chunks).strip() or "{}"

            try:
                parsed_arguments = loads(arguments_text)
            except ValueError as exception:
                raise AppException(
                    "模型返回的工具参数不是合法 JSON。",
                    error_code="invalid_llm_response",
                    details={"tool_call_id": tool_call_accumulator.tool_call_id},
                ) from exception

            if not isinstance(parsed_arguments, dict):
                raise AppException(
                    "模型返回的工具参数必须是 JSON 对象。",
                    error_code="invalid_llm_response",
                    details={"tool_call_id": tool_call_accumulator.tool_call_id},
                )

            tool_calls.append(
                LlmToolCall(
                    tool_call_id=tool_call_accumulator.tool_call_id,
                    tool_name=tool_call_accumulator.tool_name,
                    arguments=parsed_arguments,
                )
            )

        return tool_calls


class LlmClient:
    """基础大模型客户端。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._prompt_template = ChatPromptTemplate.from_messages(
            [("system", BASE_SINGLE_TURN_SYSTEM_PROMPT), ("human", "{user_message}")]
        )

    @property
    def default_model_name(self) -> str:
        """返回默认模型名，供 service 层构造兜底响应使用。"""

        return self._settings.openai_model

    async def generate_answer(self, user_message: str, model_name: str | None = None) -> str:
        """调用外部大模型生成单轮回答。"""

        prompt_value = self._prompt_template.invoke({"user_message": user_message})
        completion_result = await self.create_chat_completion(
            messages=[
                LlmInputMessage(
                    role=str(message.type),
                    content=self._normalize_message_content(str(message.content)),
                )
                for message in prompt_value.messages
            ],
            model_name=model_name,
        )
        return completion_result.content

    async def create_chat_completion(
        self,
        messages: Sequence[LlmInputMessage],
        model_name: str | None = None,
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> LlmChatCompletionResult:
        """使用统一消息结构创建一次聊天补全。"""

        request_start_time = perf_counter()
        self._log_chat_request(
            messages=messages,
            model_name=model_name,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=False,
            enable_thinking=enable_thinking,
        )
        chat_model = self._create_chat_model(
            model_name=model_name,
            is_stream=False,
            enable_thinking=enable_thinking,
        )
        runnable = chat_model.bind_tools(tools, tool_choice=tool_choice) if tools else chat_model
        llm_messages = self._build_langchain_messages(messages)

        try:
            llm_response = await runnable.ainvoke(llm_messages)
        except AppException:
            raise
        except (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            InternalServerError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        ) as exception:
            raise self._convert_openai_exception(exception) from exception

        completion_result = self._build_completion_result(
            llm_response=llm_response,
            requested_model_name=model_name,
        )
        LOGGER.info(
            (
                "LLM 请求完成：mode=non_stream model=%s duration_ms=%.2f "
                "finish_reason=%s prompt_tokens=%s completion_tokens=%s "
                "total_tokens=%s tool_call_count=%s"
            ),
            completion_result.model_name,
            (perf_counter() - request_start_time) * 1000,
            completion_result.finish_reason,
            completion_result.prompt_tokens,
            completion_result.completion_tokens,
            completion_result.total_tokens,
            len(completion_result.tool_calls),
        )
        return completion_result

    def stream_chat_completion(
        self,
        messages: Sequence[LlmInputMessage],
        model_name: str | None = None,
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[LlmChatCompletionChunk]:
        """以真实流式方式输出聊天补全增量。"""

        self._log_chat_request(
            messages=messages,
            model_name=model_name,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=True,
            enable_thinking=enable_thinking,
        )
        chat_model = self._create_chat_model(
            model_name=model_name,
            is_stream=True,
            enable_thinking=enable_thinking,
        )
        runnable = chat_model.bind_tools(tools, tool_choice=tool_choice) if tools else chat_model
        llm_messages = self._build_langchain_messages(messages)
        return self._iterate_stream_chunks(
            runnable=runnable,
            llm_messages=llm_messages,
            requested_model_name=model_name,
        )

    async def _iterate_stream_chunks(
        self,
        *,
        runnable: Any,
        llm_messages: list[BaseMessage],
        requested_model_name: str | None,
    ) -> AsyncIterator[LlmChatCompletionChunk]:
        """遍历 LangChain 流式输出并转换为统一增量结构。"""

        request_start_time = perf_counter()
        first_chunk_duration_ms: float | None = None
        chunk_count = 0
        final_finish_reason: str | None = None
        final_prompt_tokens = 0
        final_completion_tokens = 0
        final_total_tokens = 0
        resolved_model_name = requested_model_name or self._settings.openai_model

        try:
            async for llm_chunk in runnable.astream(llm_messages):
                normalized_chunk = self._build_completion_chunk(
                    llm_chunk=llm_chunk,
                    requested_model_name=requested_model_name,
                )
                chunk_count += 1
                resolved_model_name = normalized_chunk.model_name or resolved_model_name
                if first_chunk_duration_ms is None:
                    first_chunk_duration_ms = (perf_counter() - request_start_time) * 1000
                if normalized_chunk.finish_reason is not None:
                    final_finish_reason = normalized_chunk.finish_reason
                if normalized_chunk.prompt_tokens is not None:
                    final_prompt_tokens = normalized_chunk.prompt_tokens
                if normalized_chunk.completion_tokens is not None:
                    final_completion_tokens = normalized_chunk.completion_tokens
                if normalized_chunk.total_tokens is not None:
                    final_total_tokens = normalized_chunk.total_tokens
                yield normalized_chunk
            LOGGER.info(
                (
                    "LLM 请求完成：mode=stream model=%s first_chunk_ms=%s "
                    "total_ms=%.2f finish_reason=%s chunks=%s prompt_tokens=%s "
                    "completion_tokens=%s total_tokens=%s"
                ),
                resolved_model_name,
                f"{first_chunk_duration_ms:.2f}" if first_chunk_duration_ms is not None else "none",
                (perf_counter() - request_start_time) * 1000,
                final_finish_reason,
                chunk_count,
                final_prompt_tokens,
                final_completion_tokens,
                final_total_tokens,
            )
        except AppException:
            raise
        except (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            InternalServerError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        ) as exception:
            raise self._convert_openai_exception(exception) from exception

    def _create_chat_model(
        self,
        model_name: str | None = None,
        *,
        is_stream: bool = False,
        enable_thinking: bool | None = None,
    ) -> object:
        """根据环境配置创建 LangChain 聊天模型客户端。"""

        api_key = self._settings.openai_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationException(
                "未配置 OPENAI_API_KEY，无法调用大模型。",
                details={"config_key": "OPENAI_API_KEY"},
            )

        resolved_model_name = model_name or self._settings.openai_model
        provider_extra_body = self._build_provider_extra_body(
            model_name=resolved_model_name,
            is_stream=is_stream,
            enable_thinking=enable_thinking,
        )

        return init_chat_model(
            model=resolved_model_name,
            model_provider="openai",
            api_key=api_key.get_secret_value(),
            base_url=self._settings.openai_base_url or None,
            extra_body=provider_extra_body or None,
        )

    def _build_provider_extra_body(
        self,
        *,
        model_name: str,
        is_stream: bool,
        enable_thinking: bool | None,
    ) -> dict[str, Any]:
        """构造兼容提供方所需的额外请求体参数。"""

        if enable_thinking is not None:
            return {"enable_thinking": enable_thinking}

        normalized_model_name = model_name.lower()
        if normalized_model_name.startswith("qwen3") and not is_stream:
            # 部分 OpenAI 兼容网关在非流式调用 Qwen3 时要求显式关闭 thinking 模式，
            # 否则会直接返回 400 invalid_parameter_error。
            return {"enable_thinking": False}

        return {}

    def _log_chat_request(
        self,
        *,
        messages: Sequence[LlmInputMessage],
        model_name: str | None,
        tools: Sequence[LlmBindableTool] | None,
        tool_choice: str | dict[str, Any] | None,
        is_stream: bool,
        enable_thinking: bool | None,
    ) -> None:
        """记录发往大模型的完整输入，便于后续调试上下文构造。"""

        serialized_messages = [
            {
                "index": index,
                "role": message.role,
                "name": message.name,
                "content": message.content,
                "tool_call_id": message.tool_call_id,
                "tool_calls": [
                    {
                        "id": tool_call.tool_call_id,
                        "name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in message.tool_calls
                ],
            }
            for index, message in enumerate(messages)
        ]
        serialized_tool_names = []
        for tool in tools or []:
            if isinstance(tool, dict):
                function_payload = tool.get("function", {})
                tool_name = (
                    function_payload.get("name") if isinstance(function_payload, dict) else None
                )
                serialized_tool_names.append(str(tool_name or "anonymous_tool"))
            else:
                serialized_tool_names.append(str(getattr(tool, "name", tool.__class__.__name__)))

        request_payload = {
            "mode": "stream" if is_stream else "non_stream",
            "model": model_name or self._settings.openai_model,
            "tool_choice": tool_choice,
            "enable_thinking": enable_thinking,
            "tools": serialized_tool_names,
            "messages": serialized_messages,
        }
        # LOGGER.info("向 LLM 发起请求：\n %s", dumps(request_payload, ensure_ascii=False))

    def _build_langchain_messages(self, messages: Sequence[LlmInputMessage]) -> list[BaseMessage]:
        """将统一消息列表转换为 LangChain 消息对象。"""

        langchain_messages: list[BaseMessage] = []
        for message in messages:
            normalized_content = self._normalize_message_content(message.content)
            if message.role == "user":
                langchain_messages.append(
                    HumanMessage(content=normalized_content, name=message.name)
                )
            elif message.role == "assistant":
                langchain_messages.append(
                    AIMessage(
                        content=normalized_content,
                        name=message.name,
                        tool_calls=[
                            {
                                "name": tool_call.tool_name,
                                "args": tool_call.arguments,
                                "id": tool_call.tool_call_id,
                                "type": "tool_call",
                            }
                            for tool_call in message.tool_calls
                        ],
                    )
                )
            elif message.role == "system":
                langchain_messages.append(
                    SystemMessage(content=normalized_content, name=message.name)
                )
            elif message.role == "tool":
                if message.tool_call_id is None:
                    raise AppException(
                        "tool 消息缺少 tool_call_id。",
                        error_code="invalid_request",
                    )
                langchain_messages.append(
                    ToolMessage(
                        content=normalized_content,
                        tool_call_id=message.tool_call_id,
                    )
                )
            else:
                langchain_messages.append(
                    ChatMessage(role=message.role, content=normalized_content, name=message.name)
                )

        return langchain_messages

    def _build_completion_result(
        self,
        *,
        llm_response: AIMessage,
        requested_model_name: str | None,
    ) -> LlmChatCompletionResult:
        """将一次性响应转换为统一补全结果。"""

        response_text = self._extract_text_from_content(llm_response.content)
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage_metadata(
            llm_response.usage_metadata
        )
        resolved_model_name = self._resolve_model_name(
            response_metadata=llm_response.response_metadata or {},
            requested_model_name=requested_model_name,
        )
        finish_reason = str(
            (llm_response.response_metadata or {}).get("finish_reason")
            or ("tool_calls" if llm_response.tool_calls else "stop")
        )

        return LlmChatCompletionResult(
            content=response_text,
            model_name=resolved_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            tool_calls=self._extract_tool_calls(llm_response),
        )

    def _build_completion_chunk(
        self,
        *,
        llm_chunk: AIMessageChunk,
        requested_model_name: str | None,
    ) -> LlmChatCompletionChunk:
        """将 LangChain 流式消息块转换为统一增量结构。"""

        prompt_tokens, completion_tokens, total_tokens = self._extract_usage_metadata(
            llm_chunk.usage_metadata
        )

        return LlmChatCompletionChunk(
            content_delta=self._extract_text_from_content(llm_chunk.content),
            model_name=self._resolve_model_name(
                response_metadata=llm_chunk.response_metadata or {},
                requested_model_name=requested_model_name,
            ),
            prompt_tokens=prompt_tokens if total_tokens > 0 else None,
            completion_tokens=completion_tokens if total_tokens > 0 else None,
            total_tokens=total_tokens if total_tokens > 0 else None,
            finish_reason=self._extract_finish_reason(llm_chunk),
            tool_call_chunks=self._extract_tool_call_chunks(llm_chunk),
        )

    def _resolve_model_name(
        self,
        *,
        response_metadata: dict[str, Any],
        requested_model_name: str | None,
    ) -> str:
        """解析模型名，确保流式和非流式路径使用一致兜底规则。"""

        return str(
            response_metadata.get("model_name")
            or response_metadata.get("model")
            or requested_model_name
            or self._settings.openai_model
        )

    @staticmethod
    def _extract_usage_metadata(
        usage_metadata: dict[str, Any] | None,
    ) -> tuple[int, int, int]:
        """提取 token 使用量。"""

        normalized_usage = usage_metadata or {}
        prompt_tokens = int(normalized_usage.get("input_tokens", 0))
        completion_tokens = int(normalized_usage.get("output_tokens", 0))
        total_tokens = int(normalized_usage.get("total_tokens", prompt_tokens + completion_tokens))
        return prompt_tokens, completion_tokens, total_tokens

    @staticmethod
    def _extract_finish_reason(llm_chunk: AIMessageChunk) -> str | None:
        """提取流式块中的完成原因。"""

        response_metadata = llm_chunk.response_metadata or {}
        finish_reason = response_metadata.get("finish_reason")
        if finish_reason is not None:
            return str(finish_reason)

        if llm_chunk.tool_calls:
            return "tool_calls"

        return None

    @staticmethod
    def _extract_text_from_content(response_content: object) -> str:
        """从模型响应内容中提取纯文本。"""

        if isinstance(response_content, str):
            return response_content

        if isinstance(response_content, list):
            text_parts: list[str] = []
            for content_block in response_content:
                if isinstance(content_block, str) and content_block:
                    text_parts.append(content_block)
                    continue
                if isinstance(content_block, dict) and content_block.get("type") == "text":
                    text_value = content_block.get("text", "")
                    if isinstance(text_value, str) and text_value:
                        text_parts.append(text_value)
            return "".join(text_parts)

        if response_content is None:
            return ""

        return str(response_content)

    @staticmethod
    def _extract_tool_calls(llm_response: AIMessage) -> list[LlmToolCall]:
        """从模型响应中提取工具调用列表。"""

        tool_calls: list[LlmToolCall] = []
        for tool_call in llm_response.tool_calls:
            tool_arguments = tool_call.get("args", {})
            tool_calls.append(
                LlmToolCall(
                    tool_call_id=str(tool_call.get("id", "")),
                    tool_name=str(tool_call.get("name", "")),
                    arguments=tool_arguments if isinstance(tool_arguments, dict) else {},
                )
            )
        return tool_calls

    @staticmethod
    def _extract_tool_call_chunks(llm_chunk: AIMessageChunk) -> list[LlmToolCallChunk]:
        """从流式消息块中提取工具调用增量。"""

        tool_call_chunks: list[LlmToolCallChunk] = []
        for fallback_index, tool_call_chunk in enumerate(llm_chunk.tool_call_chunks):
            if not isinstance(tool_call_chunk, dict):
                continue

            tool_call_chunks.append(
                LlmToolCallChunk(
                    index=int(tool_call_chunk.get("index", fallback_index)),
                    tool_call_id=(
                        str(tool_call_chunk["id"])
                        if tool_call_chunk.get("id") is not None
                        else None
                    ),
                    tool_name=(
                        str(tool_call_chunk["name"])
                        if tool_call_chunk.get("name") is not None
                        else None
                    ),
                    arguments_chunk=(
                        str(tool_call_chunk.get("args", ""))
                        if tool_call_chunk.get("args") is not None
                        else ""
                    ),
                )
            )

        if tool_call_chunks:
            return tool_call_chunks

        # 某些兼容提供方不会返回 tool_call_chunks，而是直接在最终块给出完整 tool_calls。
        for fallback_index, tool_call in enumerate(llm_chunk.tool_calls):
            tool_arguments = tool_call.get("args", {})
            if not isinstance(tool_arguments, dict):
                tool_arguments = {}
            tool_call_chunks.append(
                LlmToolCallChunk(
                    index=fallback_index,
                    tool_call_id=str(tool_call.get("id", "")),
                    tool_name=str(tool_call.get("name", "")),
                    arguments_chunk=dumps(tool_arguments, ensure_ascii=False),
                )
            )

        return tool_call_chunks

    def _convert_openai_exception(self, exception: Exception) -> UpstreamServiceException:
        """将 OpenAI 客户端异常映射为统一业务异常。"""

        provider_body = getattr(exception, "body", None)
        provider_error = provider_body.get("error", {}) if isinstance(provider_body, dict) else {}
        provider_status_code = getattr(getattr(exception, "response", None), "status_code", None)
        provider_error_code = provider_error.get("code")
        provider_message = str(provider_error.get("message") or str(exception))

        details = {
            "provider": "openai_compatible",
            "provider_status_code": provider_status_code,
            "provider_error_code": provider_error_code,
            "provider_message": provider_message,
        }

        if isinstance(exception, AuthenticationError):
            return UpstreamServiceException(
                "LLM 提供方鉴权失败，请检查 API Key 或网关配置。",
                error_code="llm_authentication_failed",
                status_code=503,
                details=details,
            )

        if isinstance(exception, PermissionDeniedError):
            error_code = "llm_permission_denied"
            message = "LLM 提供方拒绝了当前请求。"
            if provider_error_code == "insufficient_user_quota":
                error_code = "llm_quota_exceeded"
                message = "LLM 提供方额度不足，请更换可用 Key 或充值后重试。"
            return UpstreamServiceException(
                message,
                error_code=error_code,
                status_code=503,
                details=details,
            )

        if isinstance(exception, RateLimitError):
            return UpstreamServiceException(
                "LLM 提供方触发限流，请稍后重试。",
                error_code="llm_rate_limited",
                status_code=429,
                details=details,
            )

        if isinstance(exception, (BadRequestError, UnprocessableEntityError)):
            return UpstreamServiceException(
                "发送到 LLM 提供方的请求参数不合法。",
                error_code="llm_bad_request",
                status_code=400,
                details=details,
            )

        if isinstance(exception, (APIConnectionError, APITimeoutError, InternalServerError)):
            return UpstreamServiceException(
                "LLM 提供方暂时不可用，请稍后重试。",
                error_code="llm_unavailable",
                status_code=503,
                details=details,
            )

        return UpstreamServiceException(
            "调用 LLM 提供方时发生未知错误。",
            details=details,
        )

    @staticmethod
    def _normalize_message_content(content: str) -> str:
        """规范化消息文本，避免将纯空白字符直接传给模型。"""

        return content.strip()
