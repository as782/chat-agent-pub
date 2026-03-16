"""LLM 客户端模块。

负责统一封装对外部大模型的调用，避免 service 层直接依赖第三方 SDK。
当前阶段仅负责基础文本问答，不负责流式输出、工具调用和多模型路由。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.exceptions import ConfigurationException


@dataclass(slots=True)
class LlmChatCompletionResult:
    """大模型对话结果。"""

    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str


class LlmClient:
    """基础大模型客户端。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是最小可用 Agent 后端中的基础问答模块，需要简洁、准确地回答用户。"),
                ("human", "{user_message}"),
            ]
        )

    async def generate_answer(self, user_message: str, model_name: str | None = None) -> str:
        """调用外部大模型生成单轮回答。"""

        prompt_value = self._prompt_template.invoke({"user_message": user_message})
        completion_result = await self.create_chat_completion(
            messages=[
                (
                    str(message.type),
                    self._normalize_message_content(str(message.content)),
                )
                for message in prompt_value.messages
            ],
            model_name=model_name,
        )
        return completion_result.content

    async def create_chat_completion(
        self,
        messages: Sequence[tuple[str, str]],
        model_name: str | None = None,
    ) -> LlmChatCompletionResult:
        """使用标准消息列表创建一次聊天补全。"""

        chat_model = self._create_chat_model(model_name=model_name)
        llm_messages = self._build_langchain_messages(messages)
        llm_response = await chat_model.ainvoke(llm_messages)
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
        finish_reason = str(response_metadata.get("finish_reason", "stop"))

        return LlmChatCompletionResult(
            content=response_text,
            model_name=resolved_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    def _create_chat_model(self, model_name: str | None = None) -> ChatOpenAI:
        """根据环境配置创建 ChatOpenAI 客户端。"""

        api_key = self._settings.openai_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationException(
                "未配置 OPENAI_API_KEY，无法调用大模型。",
                details={"config_key": "OPENAI_API_KEY"},
            )

        return ChatOpenAI(
            model=model_name or self._settings.openai_model,
            api_key=api_key.get_secret_value(),
            base_url=self._settings.openai_base_url or None,
        )

    def _build_langchain_messages(self, messages: Sequence[tuple[str, str]]) -> list[BaseMessage]:
        """将标准消息列表转换为 LangChain 消息对象。"""

        langchain_messages: list[BaseMessage] = []
        for role, content in messages:
            normalized_content = self._normalize_message_content(content)
            if role == "user":
                langchain_messages.append(HumanMessage(content=normalized_content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=normalized_content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=normalized_content))
            else:
                langchain_messages.append(ChatMessage(role=role, content=normalized_content))

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
    def _normalize_message_content(content: str) -> str:
        """规范化消息文本，避免将空白字符直接传给模型。"""

        return content.strip()
