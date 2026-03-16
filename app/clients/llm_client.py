"""LLM 客户端模块。
负责统一封装对外部大模型的调用，避免 service 层直接依赖第三方 SDK。
当前阶段支持基础文本问答、工具调用与错误映射，不负责多模型路由和复杂重试策略。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
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

from app.core.config import get_settings
from app.core.exceptions import AppException, ConfigurationException, UpstreamServiceException


@dataclass(slots=True)
class LlmToolCall:
    """大模型返回的工具调用。"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class LlmInputMessage:
    """统一的输入消息模型。"""

    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[LlmToolCall] = field(default_factory=list)


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


class LlmClient:
    """基础大模型客户端。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是最小可用 Agent 后端中的基础问答模块，需要简洁、准确地回答用户。",
                ),
                ("human", "{user_message}"),
            ]
        )

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
        tools: Sequence[BaseTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LlmChatCompletionResult:
        """使用统一消息结构创建一次聊天补全。"""

        chat_model = self._create_chat_model(model_name=model_name)
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

        response_text = self._extract_text_from_response(llm_response)
        usage_metadata = llm_response.usage_metadata or {}
        response_metadata = llm_response.response_metadata or {}

        prompt_tokens = int(usage_metadata.get("input_tokens", 0))
        completion_tokens = int(usage_metadata.get("output_tokens", 0))
        total_tokens = int(usage_metadata.get("total_tokens", prompt_tokens + completion_tokens))
        resolved_model_name = str(
            response_metadata.get("model_name")
            or response_metadata.get("model")
            or model_name
            or self._settings.openai_model
        )
        finish_reason = str(
            response_metadata.get("finish_reason")
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

    def _create_chat_model(self, model_name: str | None = None) -> object:
        """根据环境配置创建 LangChain 聊天模型客户端。"""

        api_key = self._settings.openai_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationException(
                "未配置 OPENAI_API_KEY，无法调用大模型。",
                details={"config_key": "OPENAI_API_KEY"},
            )

        return init_chat_model(
            model=model_name or self._settings.openai_model,
            model_provider="openai",
            api_key=api_key.get_secret_value(),
            base_url=self._settings.openai_base_url or None,
        )

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

    @staticmethod
    def _extract_text_from_response(llm_response: AIMessage) -> str:
        """从模型响应中提取纯文本内容。"""

        response_content = llm_response.content
        if isinstance(response_content, str):
            return response_content.strip()

        if isinstance(response_content, list):
            text_parts: list[str] = []
            for content_block in response_content:
                if isinstance(content_block, dict) and content_block.get("type") == "text":
                    text_value = content_block.get("text", "")
                    if isinstance(text_value, str) and text_value:
                        text_parts.append(text_value)
            if text_parts:
                return "\n".join(text_parts).strip()

        return str(response_content).strip()

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
        """规范化消息文本，避免将空白字符直接传给模型。"""

        return content.strip()
