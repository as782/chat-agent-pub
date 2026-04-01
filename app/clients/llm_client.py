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

from langchain.chat_models import init_chat_model
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

    def _create_chat_model(
        self,
        model_name: str | None = None,
        *,
        is_stream: bool = False,
        enable_thinking: bool | None = None,
    ) -> BaseChatModel:
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

    def create_runnable(
        self,
        *,
        model_name: str | None = None,
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        is_stream: bool = False,
        enable_thinking: bool | None = None,
    ) -> Any:
        """创建一个可直接执行的 LangChain Runnable 对象。"""

        chat_model = self._create_chat_model(
            model_name=model_name,
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
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> AIMessage:
        """使用统一消息结构创建一次聊天补全。"""

        request_start_time = perf_counter()
        # 记录输入
        self._log_chat_request(
            messages=messages,
            model_name=model_name,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=False,
            enable_thinking=enable_thinking,
        )
        
        runnable = self.create_runnable(
            model_name=model_name,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=False,
            enable_thinking=enable_thinking,
        )
        llm_messages = self._build_langchain_messages(messages)

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
            raise self._convert_openai_exception(exception) from exception

        LOGGER.info(
            (
                "LLM 请求完成：mode=non_stream model=%s duration_ms=%.2f "
                "tool_call_count=%d"
            ),
            model_name or self._settings.openai_model,
            (perf_counter() - request_start_time) * 1000,
            len(completion_result.tool_calls),
        )
        return completion_result

    async def _iterate_stream_chunks(
        self,
        *,
        runnable: Any,
        llm_messages: list[BaseMessage],
        requested_model_name: str | None,
    ) -> AsyncIterator[AIMessageChunk]:
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
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        ) as exception:
            raise self._convert_openai_exception(exception) from exception

    def stream_chat_completion(
        self,
        messages: Sequence[LlmInputMessage],
        model_name: str | None = None,
        tools: Sequence[LlmBindableTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[AIMessageChunk]:
        """以真实流式方式输出聊天补全增量。"""

        self._log_chat_request(
            messages=messages,
            model_name=model_name,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=True,
            enable_thinking=enable_thinking,
        )
        runnable = self.create_runnable(
            model_name=model_name,
            tools=tools,
            tool_choice=tool_choice,
            is_stream=True,
            enable_thinking=enable_thinking,
        )
        llm_messages = self._build_langchain_messages(messages)
        return self._iterate_stream_chunks(
            runnable=runnable,
            llm_messages=llm_messages,
            requested_model_name=model_name,
        )

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

        LOGGER.info("向 LLM 发起请求：\n %s", dumps({
            "mode": "stream" if is_stream else "non_stream",
            "model": model_name or self._settings.openai_model,
            "tool_choice": tool_choice,
            "enable_thinking": enable_thinking,
            "tools": serialized_tool_names,
            "messages": serialized_messages,
        }, ensure_ascii=False))

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

