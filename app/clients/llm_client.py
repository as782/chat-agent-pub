"""LLM 客户端模块。
负责统一封装对外部大模型的调用，避免 service 层直接依赖第三方 SDK。
当前阶段支持普通补全、真实流式输出与工具调用，不负责多模型路由和复杂重试策略。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from json import dumps
from time import perf_counter
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
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
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from app.agent.prompts import BASE_SINGLE_TURN_SYSTEM_PROMPT
from app.core.config import MONITOR_NETWORK_OPENAI_PROXY_PATH, get_settings
from app.core.exceptions import AppException, ConfigurationException, UpstreamServiceException
from app.core.logger import get_logger

LOGGER = get_logger(__name__)

LlmBindableTool = BaseTool | dict[str, Any]


class _PathRewriteTransport(httpx.BaseTransport):
    """Rewrite non-standard monitor proxy completion paths for sync OpenAI SDK calls."""

    def __init__(self, *, source_paths: tuple[str, ...], target_path: str) -> None:
        self._source_paths = source_paths
        self._target_path = target_path
        self._transport = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path in self._source_paths:
            request = httpx.Request(
                method=request.method,
                url=request.url.copy_with(path=self._target_path),
                headers=request.headers,
                stream=request.stream,
                extensions=request.extensions,
            )
        return self._transport.handle_request(request)

    def close(self) -> None:
        self._transport.close()


class _AsyncPathRewriteTransport(httpx.AsyncBaseTransport):
    """Rewrite non-standard monitor proxy completion paths for async OpenAI SDK calls."""

    def __init__(self, *, source_paths: tuple[str, ...], target_path: str) -> None:
        self._source_paths = source_paths
        self._target_path = target_path
        self._transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path in self._source_paths:
            request = httpx.Request(
                method=request.method,
                url=request.url.copy_with(path=self._target_path),
                headers=request.headers,
                stream=request.stream,
                extensions=request.extensions,
            )
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


def _patch_langchain_openai_reasoning_content_support() -> None:
    """Preserve third-party OpenAI-compatible reasoning fields in LangChain messages."""

    try:
        import langchain_openai.chat_models.base as langchain_openai_base
    except Exception:  # pragma: no cover - optional dependency import guard
        return

    if getattr(langchain_openai_base, "_chat_agent_reasoning_content_patched", False):
        return

    original_convert_dict_to_message = langchain_openai_base._convert_dict_to_message
    original_convert_delta_to_message_chunk = (
        langchain_openai_base._convert_delta_to_message_chunk
    )

    def patched_convert_dict_to_message(_dict: dict[str, Any]) -> BaseMessage:
        message = original_convert_dict_to_message(_dict)
        reasoning_content = _dict.get("reasoning_content")
        if isinstance(message, AIMessage) and isinstance(reasoning_content, str):
            message.additional_kwargs["reasoning_content"] = reasoning_content
        return message

    def patched_convert_delta_to_message_chunk(
        _dict: dict[str, Any],
        default_class: type[BaseMessage],
    ) -> BaseMessage:
        message_chunk = original_convert_delta_to_message_chunk(_dict, default_class)
        reasoning_content = _dict.get("reasoning_content")
        if isinstance(message_chunk, AIMessageChunk) and isinstance(reasoning_content, str):
            message_chunk.additional_kwargs["reasoning_content"] = reasoning_content
        return message_chunk

    langchain_openai_base._convert_dict_to_message = patched_convert_dict_to_message
    langchain_openai_base._convert_delta_to_message_chunk = (
        patched_convert_delta_to_message_chunk
    )
    langchain_openai_base._chat_agent_reasoning_content_patched = True


_patch_langchain_openai_reasoning_content_support()


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

        return self._settings.resolved_openai_model

    @staticmethod
    def extract_llm_tool_calls(llm_response: Any) -> list[LlmToolCall]:
        """从模型响应中提取统一工具调用列表。"""

        tool_calls: list[LlmToolCall] = []
        raw_tool_calls = getattr(llm_response, "tool_calls", []) or []
        
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                # Native LangChain tool call dict
                tool_calls.append(
                    LlmToolCall(
                        tool_call_id=str(tc.get("id", "")),
                        tool_name=str(tc.get("name", "")),
                        arguments=tc.get("args", {}) if isinstance(tc.get("args"), dict) else {},
                    )
                )
            else:
                # Internal ExecutedToolCall or similar object
                tool_calls.append(
                    LlmToolCall(
                        tool_call_id=str(getattr(tc, "tool_call_id", getattr(tc, "id", ""))),
                        tool_name=str(getattr(tc, "tool_name", getattr(tc, "name", ""))),
                        arguments=getattr(tc, "arguments", getattr(tc, "args", {})),
                    )
                )
        return tool_calls

    @staticmethod
    def _normalize_message_content(content: str) -> str:
        """规范化消息文本，避免将纯空白字符直接传给模型。"""

        return content.strip()

    def _normalize_outbound_messages(
        self,
        messages: Sequence[LlmInputMessage],
        *,
        model_name: str | None,
    ) -> list[LlmInputMessage]:
        """Normalize provider-bound messages before sending them upstream."""

        _ = model_name
        normalized_messages = list(messages)
        merged_messages: list[LlmInputMessage] = []
        pending_system_messages: list[LlmInputMessage] = []

        def flush_pending_system_messages() -> None:
            if not pending_system_messages:
                return
            if len(pending_system_messages) == 1:
                merged_messages.append(pending_system_messages[0])
            else:
                merged_content = "\n\n".join(
                    content
                    for content in (
                        self._normalize_message_content(message.content)
                        for message in pending_system_messages
                    )
                    if content
                )
                merged_messages.append(
                    LlmInputMessage(
                        role="system",
                        content=merged_content,
                        name=pending_system_messages[0].name,
                    )
                )
            pending_system_messages.clear()

        for message in normalized_messages:
            if message.role == "system":
                pending_system_messages.append(message)
                continue
            flush_pending_system_messages()
            merged_messages.append(message)

        flush_pending_system_messages()
        return merged_messages

    def _build_provider_extra_body(
        self,
        *,
        model_name: str,
        is_stream: bool,
        enable_thinking: bool | None,
    ) -> dict[str, Any]:
        """构造兼容提供方所需的额外请求体参数。"""

        resolved_enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else self._settings.openai_enable_thinking
        )

        normalized_model_name = model_name.lower()
        if resolved_enable_thinking is None and normalized_model_name.startswith("qwen3") and not is_stream:
            # 部分 OpenAI 兼容网关在非流式调用 Qwen3 时要求显式关闭 thinking 模式，
            # 否则会直接返回 400 invalid_parameter_error。
            resolved_enable_thinking = False

        if resolved_enable_thinking is None:
            return {}

        return {
            "enable_thinking": resolved_enable_thinking,
            "chat_template_kwargs": {"enable_thinking": resolved_enable_thinking},
        }

    def _resolve_base_url(self, base_url: str | None) -> str | None:
        """解析本次请求最终使用的 base_url。"""

        return (
            base_url
            if base_url is not None
            else (self._settings.resolved_openai_base_url or None)
        )

    def _resolve_timeout_seconds(self, timeout_seconds: float | None) -> float:
        """解析本次请求最终使用的超时时间。"""

        return timeout_seconds if timeout_seconds is not None else self._settings.openai_timeout_seconds

    @staticmethod
    def _build_httpx_timeout(connect_timeout_seconds: float) -> httpx.Timeout:
        return httpx.Timeout(None, connect=connect_timeout_seconds)

    def _build_langchain_messages(
        self,
        messages: Sequence[LlmInputMessage],
        *,
        model_name: str | None = None,
    ) -> list[BaseMessage]:
        """将统一消息列表转换为 LangChain 消息对象。"""

        langchain_messages: list[BaseMessage] = []
        normalized_messages = self._normalize_outbound_messages(messages, model_name=model_name)
        for message in normalized_messages:
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

    def _create_chat_model(
        self,
        model_name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        is_stream: bool = False,
        enable_thinking: bool | None = None,
    ) -> BaseChatModel:
        """根据环境配置创建 LangChain 聊天模型客户端。"""

        resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = self._settings.resolved_openai_api_key_value
        if resolved_api_key is None or not resolved_api_key.strip():
            raise ConfigurationException(
                "未配置 OPENAI_API_KEY，无法调用大模型。",
                details={"config_key": "OPENAI_API_KEY"},
            )

        resolved_model_name = model_name or self._settings.resolved_openai_model
        resolved_base_url = self._resolve_base_url(base_url)
        resolved_timeout_seconds = self._resolve_timeout_seconds(timeout_seconds)
        provider_extra_body = self._build_provider_extra_body(
            model_name=resolved_model_name,
            is_stream=is_stream,
            enable_thinking=enable_thinking,
        )
        sync_http_client = None
        async_http_client = None
        if self._should_use_monitor_network_openai_proxy(resolved_base_url):
            sync_http_client, async_http_client = self._build_monitor_network_http_clients()

        return ChatOpenAI(
            model=resolved_model_name,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=self._build_httpx_timeout(resolved_timeout_seconds),
            extra_body=provider_extra_body or None,
            http_client=sync_http_client,
            http_async_client=async_http_client,
        )

    def _should_use_monitor_network_openai_proxy(self, resolved_base_url: str | None) -> bool:
        """Return whether the current call should rewrite /chat/completions to monitor-completions."""

        return (
            self._settings.use_monitor_network_development_upstreams
            and resolved_base_url == self._settings.resolved_openai_base_url
        )

    def _build_monitor_network_http_clients(self) -> tuple[httpx.Client, httpx.AsyncClient]:
        """Build sync/async httpx clients that rewrite the completion path for the monitor proxy."""

        source_paths = ("/chat/completions", "/v1/chat/completions")
        return (
            httpx.Client(
                transport=_PathRewriteTransport(
                    source_paths=source_paths,
                    target_path=MONITOR_NETWORK_OPENAI_PROXY_PATH,
                )
            ),
            httpx.AsyncClient(
                transport=_AsyncPathRewriteTransport(
                    source_paths=source_paths,
                    target_path=MONITOR_NETWORK_OPENAI_PROXY_PATH,
                )
            ),
        )

    def create_runnable(
        self,
        *,
        messages: Sequence[LlmInputMessage] | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        is_stream: bool = False,
        enable_thinking: bool | None = None,
        log_format: str = "default",
    ) -> Any:
        """创建一个可直接执行的 LangChain Runnable 对象。"""

        resolved_base_url = self._resolve_base_url(base_url)
        resolved_timeout_seconds = self._resolve_timeout_seconds(timeout_seconds)
        normalized_messages = self._normalize_outbound_messages(
            messages or [],
            model_name=model_name or self._settings.resolved_openai_model,
        )
        self._log_chat_request(
            messages=normalized_messages,
            model_name=model_name,
            base_url=resolved_base_url,
            timeout_seconds=resolved_timeout_seconds,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=is_stream,
            enable_thinking=enable_thinking,
            log_format=log_format,
        )

        chat_model = self._create_chat_model(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            is_stream=is_stream,
            enable_thinking=enable_thinking,
        )
        if tools:
            return chat_model.bind_tools(tools, tool_choice=tool_choice)
        return chat_model

    async def generate_answer(self, user_message: str, model_name: str | None = None) -> str:
        """调用LlmClient配置的大模型生成单轮回答。"""

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
        # AIMessage.content 可能是字符串或 ContentBlock 列表
        if isinstance(completion_result.content, str):
            return completion_result.content
        return str(completion_result.content)

    async def create_chat_completion(
        self,
        messages: Sequence[LlmInputMessage],
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
        log_format: str = "default",
    ) -> AIMessage:
        """使用统一消息结构创建一次聊天补全。"""

        request_start_time = perf_counter()
        resolved_base_url = self._resolve_base_url(base_url)
        resolved_timeout_seconds = self._resolve_timeout_seconds(timeout_seconds)
        normalized_messages = self._normalize_outbound_messages(
            messages,
            model_name=model_name or self._settings.resolved_openai_model,
        )
        runnable = self.create_runnable(
            messages=normalized_messages,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=resolved_timeout_seconds,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=False,
            enable_thinking=enable_thinking,
            log_format=log_format,
        )
        llm_messages = self._build_langchain_messages(
            normalized_messages,
            model_name=model_name or self._settings.resolved_openai_model,
        )

        try:
            completion_result = await runnable.ainvoke(llm_messages)
        except AppException:
            raise
        except (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            InternalServerError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        ) as exception:
            LOGGER.warning(
                (
                    "LLM 璇锋眰澶辫触锛歮ode=non_stream model=%s duration_ms=%.2f "
                    "base_url=%s connect_timeout_seconds=%.2f error_type=%s"
                ),
                model_name or self._settings.resolved_openai_model,
                (perf_counter() - request_start_time) * 1000,
                resolved_base_url or "default",
                resolved_timeout_seconds,
                type(exception).__name__,
            )
            raise self._convert_openai_exception(exception) from exception

        LOGGER.info(
            (
                "LLM 请求完成：mode=non_stream model=%s duration_ms=%.2f "
                "tool_call_count=%d base_url=%s connect_timeout_seconds=%.2f"
            ),
            model_name or self._settings.resolved_openai_model,
            (perf_counter() - request_start_time) * 1000,
            len(completion_result.tool_calls),
            resolved_base_url or "default",
            resolved_timeout_seconds,
        )
        return completion_result

    async def _iterate_stream_chunks(
        self,
        *,
        runnable: Any,
        llm_messages: list[BaseMessage],
        requested_model_name: str | None,
        requested_base_url: str | None,
        timeout_seconds: float,
    ) -> AsyncIterator[AIMessageChunk]:
        """遍历 LangChain 流式输出并转换为统一增量结构。"""

        request_start_time = perf_counter()
        first_chunk_duration_ms: float | None = None
        chunk_count = 0
        final_finish_reason: str | None = None
        final_prompt_tokens = 0
        final_completion_tokens = 0
        final_total_tokens = 0
        resolved_model_name = requested_model_name or self._settings.resolved_openai_model

        try:
            async for llm_chunk in runnable.astream(llm_messages):
                if not isinstance(llm_chunk, AIMessageChunk):
                    continue
                chunk_count += 1
                if first_chunk_duration_ms is None:
                    first_chunk_duration_ms = (perf_counter() - request_start_time) * 1000
                yield llm_chunk
            LOGGER.info(
                (
                    "LLM 请求完成：mode=stream model=%s first_chunk_ms=%s "
                    "total_ms=%.2f finish_reason=%s chunks=%s prompt_tokens=%s "
                    "completion_tokens=%s total_tokens=%s base_url=%s connect_timeout_seconds=%.2f"
                ),
                resolved_model_name,
                f"{first_chunk_duration_ms:.2f}" if first_chunk_duration_ms is not None else "none",
                (perf_counter() - request_start_time) * 1000,
                final_finish_reason,
                chunk_count,
                final_prompt_tokens,
                final_completion_tokens,
                final_total_tokens,
                requested_base_url or "default",
                timeout_seconds,
            )
        except AppException:
            raise
        except (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            InternalServerError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        ) as exception:
            LOGGER.warning(
                (
                    "LLM 璇锋眰澶辫触锛歮ode=stream model=%s duration_ms=%.2f "
                    "base_url=%s connect_timeout_seconds=%.2f error_type=%s"
                ),
                resolved_model_name,
                (perf_counter() - request_start_time) * 1000,
                requested_base_url or "default",
                timeout_seconds,
                type(exception).__name__,
            )
            raise self._convert_openai_exception(exception) from exception

    def stream_chat_completion(
        self,
        messages: Sequence[LlmInputMessage],
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
        log_format: str = "default",
    ) -> AsyncIterator[AIMessageChunk]:
        """以真实流式方式输出聊天补全增量。"""

        resolved_base_url = self._resolve_base_url(base_url)
        resolved_timeout_seconds = self._resolve_timeout_seconds(timeout_seconds)
        normalized_messages = self._normalize_outbound_messages(
            messages,
            model_name=model_name or self._settings.resolved_openai_model,
        )
        runnable = self.create_runnable(
            messages=normalized_messages,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=resolved_timeout_seconds,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=True,
            enable_thinking=enable_thinking,
            log_format=log_format,
        )
        llm_messages = self._build_langchain_messages(
            normalized_messages,
            model_name=model_name or self._settings.resolved_openai_model,
        )
        return self._iterate_stream_chunks(
            runnable=runnable,
            llm_messages=llm_messages,
            requested_model_name=model_name,
            requested_base_url=resolved_base_url,
            timeout_seconds=resolved_timeout_seconds,
        )

    def _log_chat_request(
        self,
        *,
        messages: Sequence[LlmInputMessage],
        model_name: str | None,
        base_url: str | None,
        timeout_seconds: float,
        tools: Sequence[LlmBindableTool] | None,
        tool_choice: str | dict[str, Any] | None,
        is_stream: bool,
        enable_thinking: bool | None,
        log_format: str,
    ) -> None:
        """记录发往大模型的完整输入，便于后续调试上下文构造。"""

        # 构造完整 endpoint 路径：如果 base_url 已经包含 /v1，则只追加 /chat/completions
        resolved_model_name = model_name or self._settings.resolved_openai_model
        provider_extra_body = self._build_provider_extra_body(
            model_name=resolved_model_name,
            is_stream=is_stream,
            enable_thinking=enable_thinking,
        )
        endpoint_path = "/chat/completions"
        if base_url:
            normalized_base_url = base_url.rstrip("/")
            if normalized_base_url.endswith("/v1"):
                full_url = normalized_base_url + endpoint_path
            else:
                full_url = normalized_base_url + "/v1" + endpoint_path
        else:
            full_url = "(default base_url)/v1" + endpoint_path

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

        if log_format == "curl":
            LOGGER.info(
                self._format_chat_request_log_as_curl(
                    full_url=full_url,
                    mode="stream" if is_stream else "non_stream",
                    model_name=resolved_model_name,
                    base_url=base_url or "default",
                    timeout_seconds=timeout_seconds,
                    tool_choice=tool_choice,
                    enable_thinking=enable_thinking,
                    extra_body=provider_extra_body or None,
                    tools=serialized_tool_names,
                    messages=messages,
                )
            )
            return

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
        LOGGER.info(
            "向 LLM 发起请求：\n完整URL: %s\n参数: %s",
            full_url,
            dumps(
                {
                    "mode": "stream" if is_stream else "non_stream",
                    "model": resolved_model_name,
                    "base_url": base_url or "default",
                    "connect_timeout_seconds": timeout_seconds,
                    "tool_choice": tool_choice,
                    "enable_thinking": enable_thinking,
                    "extra_body": provider_extra_body or None,
                    "tools": serialized_tool_names,
                    "messages": serialized_messages,
                },
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def _format_chat_request_log_as_curl(
        *,
        full_url: str,
        mode: str,
        model_name: str,
        base_url: str,
        timeout_seconds: float,
        tool_choice: str | dict[str, Any] | None,
        enable_thinking: bool | None,
        extra_body: dict[str, Any] | None,
        tools: Sequence[str],
        messages: Sequence[LlmInputMessage],
    ) -> str:
        """构造可直接复制执行的 curl 风格 LLM 请求日志。"""

        request_payload = LlmClient._build_log_request_payload(
            mode=mode,
            model_name=model_name,
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
            tools=tools,
            messages=messages,
        )
        pretty_payload = dumps(request_payload, ensure_ascii=False, indent=2)
        curl_command = LlmClient._build_curl_command(
            full_url=full_url,
            pretty_payload=pretty_payload,
        )
        return "\n".join(
            [
                "向 LLM 发起请求：",
                f"模式: {mode}",
                f"模型: {model_name}",
                f"base_url: {base_url}",
                f"connect_timeout_seconds: {timeout_seconds}",
                "curl 复现命令:",
                curl_command,
            ]
        )

    @staticmethod
    def _build_log_request_payload(
        *,
        mode: str,
        model_name: str,
        tool_choice: str | dict[str, Any] | None,
        enable_thinking: bool | None,
        extra_body: dict[str, Any] | None,
        tools: Sequence[str],
        messages: Sequence[LlmInputMessage],
    ) -> dict[str, Any]:
        """构造贴近真实上游请求体的日志 payload。"""

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                LlmClient._serialize_message_for_log(message)
                for message in messages
            ],
            "stream": mode == "stream",
        }
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if extra_body:
            payload.update(extra_body)
        if tools:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    @staticmethod
    def _serialize_message_for_log(message: LlmInputMessage) -> dict[str, Any]:
        """把内部消息模型转换为更贴近 OpenAI 接口的日志格式。"""

        serialized_message: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.name:
            serialized_message["name"] = message.name
        if message.tool_call_id:
            serialized_message["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            serialized_message["tool_calls"] = [
                {
                    "id": tool_call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return serialized_message

    @staticmethod
    def _build_curl_command(*, full_url: str, pretty_payload: str) -> str:
        """构造 bash 兼容的 curl 命令，便于复制复现。"""

        return "\n".join(
            [
                f"curl --location --request POST {LlmClient._shell_single_quote(full_url)} \\",
                "  --header 'Content-Type: application/json' \\",
                f"  --data-raw {LlmClient._shell_single_quote(pretty_payload)}",
            ]
        )

    def _format_log_value(value: object) -> str:
        """稳定格式化复杂值，避免日志里出现难读的 Python 表示。"""

        if isinstance(value, str):
            return value
        return dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _shell_single_quote(text: str) -> str:
        """用单引号安全包裹 shell 文本，允许内容中包含单引号。"""

        return "'" + text.replace("'", "'\"'\"'") + "'"

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

        if isinstance(exception, NotFoundError):
            error_code = "llm_resource_not_found"
            message = "LLM 提供方未找到请求的资源。"
            if provider_error_code == "model_not_found":
                error_code = "llm_model_not_found"
                message = "指定的模型不存在，或当前账号/网关无权访问该模型。"
            return UpstreamServiceException(
                message,
                error_code=error_code,
                status_code=404,
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

