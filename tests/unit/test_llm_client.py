"""LLM 客户端单元测试。"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import PermissionDeniedError
from pytest import MonkeyPatch

from app.clients.llm_client import LlmClient
from app.core.exceptions import ConfigurationException, UpstreamServiceException


class FakeChatModel:
    """用于测试的假聊天模型。"""

    last_init_kwargs: dict[str, object] = {}
    last_tool_choice: str | dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        self.__class__.last_init_kwargs = kwargs

    def bind_tools(
        self,
        _: list[object],
        *,
        tool_choice: str | dict[str, object] | None = None,
        **__: object,
    ) -> FakeChatModel:
        """记录工具绑定参数并返回自身。"""

        self.__class__.last_tool_choice = tool_choice
        return self

    async def ainvoke(self, _: object) -> AIMessage:
        """返回稳定的假模型响应。"""

        return AIMessage(content="模拟大模型回答")


class FakePermissionDeniedModel(FakeChatModel):
    """用于测试额度不足错误的假模型。"""

    async def ainvoke(self, _: object) -> AIMessage:
        """抛出模拟的额度不足异常。"""

        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(status_code=403, request=request)
        raise PermissionDeniedError(
            "quota exceeded",
            response=response,
            body={
                "error": {
                    "message": "quota exceeded",
                    "code": "insufficient_user_quota",
                }
            },
        )


@pytest.mark.asyncio
async def test_llm_client_builds_chat_model_from_settings(monkeypatch: MonkeyPatch) -> None:
    """验证 LLM 客户端会按配置组装 LangChain 模型初始化参数。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    answer = await llm_client.generate_answer("你好")

    assert answer == "模拟大模型回答"
    assert FakeChatModel.last_init_kwargs["model"] == "unit-test-model"
    assert FakeChatModel.last_init_kwargs["model_provider"] == "openai"
    assert FakeChatModel.last_init_kwargs["api_key"] == "unit-test-key"
    assert FakeChatModel.last_init_kwargs["base_url"] == "https://example.com/v1"


@pytest.mark.asyncio
async def test_llm_client_maps_quota_error_to_upstream_exception(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证额度不足错误会被映射为明确的上游服务异常。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakePermissionDeniedModel)

    llm_client = LlmClient()

    with pytest.raises(UpstreamServiceException) as exception_info:
        await llm_client.generate_answer("你好")

    assert exception_info.value.error_code == "llm_quota_exceeded"
    assert exception_info.value.status_code == 503


def test_llm_client_raises_configuration_error_when_api_key_missing(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证未配置 API Key 时会抛出明确异常。"""

    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    llm_client = LlmClient()

    with pytest.raises(ConfigurationException):
        llm_client._create_chat_model()
