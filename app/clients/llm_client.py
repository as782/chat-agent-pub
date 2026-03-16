"""LLM 客户端模块。

负责统一封装对外部大模型的调用，避免 service 层直接依赖第三方 SDK。
当前阶段仅负责基础单轮问答，不负责流式输出、工具调用和多模型路由。
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.exceptions import ConfigurationException


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

    async def generate_answer(self, user_message: str) -> str:
        """调用外部大模型生成单轮回答。"""

        prompt_value = self._prompt_template.invoke({"user_message": user_message})
        chat_model = self._create_chat_model()
        llm_response = await chat_model.ainvoke(prompt_value.messages)
        return self._extract_text_from_response(llm_response)

    def _create_chat_model(self) -> ChatOpenAI:
        """根据环境配置创建 ChatOpenAI 客户端。"""

        api_key = self._settings.openai_api_key
        if api_key is None:
            raise ConfigurationException(
                "未配置 OPENAI_API_KEY，无法调用大模型。",
                details={"config_key": "OPENAI_API_KEY"},
            )

        return ChatOpenAI(
            model=self._settings.openai_model,
            api_key=api_key.get_secret_value(),
            base_url=self._settings.openai_base_url or None,
            temperature=self._settings.openai_temperature,
        )

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
