"""LLM 客户端单元测试。"""

import pytest
from langchain_core.messages import AIMessage
from pytest import MonkeyPatch

from app.clients.llm_client import LlmClient
from app.core.exceptions import ConfigurationException


class FakeChatOpenAI:
    """用于测试的假 ChatOpenAI 客户端。"""

    last_init_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        self.__class__.last_init_kwargs = kwargs

    async def ainvoke(self, _: object) -> AIMessage:
        """返回稳定的假模型响应。"""

        return AIMessage(content="模拟大模型回答")


@pytest.mark.asyncio
async def test_llm_client_builds_chat_model_from_settings(monkeypatch: MonkeyPatch) -> None:
    """验证 LLM 客户端会按配置组装 ChatOpenAI 参数。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.ChatOpenAI", FakeChatOpenAI)

    llm_client = LlmClient()
    answer = await llm_client.generate_answer("你好")

    assert answer == "模拟大模型回答"
    assert FakeChatOpenAI.last_init_kwargs["model"] == "unit-test-model"
    assert FakeChatOpenAI.last_init_kwargs["api_key"] == "unit-test-key"
    assert FakeChatOpenAI.last_init_kwargs["base_url"] == "https://example.com/v1"


def test_llm_client_raises_configuration_error_when_api_key_missing(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证未配置 API Key 时会抛出明确异常。"""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    llm_client = LlmClient()

    with pytest.raises(ConfigurationException):
        llm_client._create_chat_model()
