"""LLM 客户端单元测试。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from openai import PermissionDeniedError
from pytest import MonkeyPatch

from app.clients.llm_client import LlmChatCompletionChunk, LlmClient, LlmInputMessage
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

    async def astream(self, _: object) -> AsyncIterator[AIMessageChunk]:
        """返回稳定的假流式响应。"""

        yield AIMessageChunk(content="模拟", response_metadata={"model_name": "unit-test-model"})
        yield AIMessageChunk(
            content="流式回答",
            response_metadata={"model_name": "unit-test-model"},
        )
        yield AIMessageChunk(
            content="",
            response_metadata={"model_name": "unit-test-model", "finish_reason": "stop"},
            usage_metadata={"input_tokens": 2, "output_tokens": 4, "total_tokens": 6},
        )


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
    assert FakeChatModel.last_init_kwargs["extra_body"] is None


@pytest.mark.asyncio
async def test_llm_client_disables_qwen3_thinking_for_non_stream_calls(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证非流式调用 Qwen3 时会自动关闭 thinking 模式。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3-32b")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[LlmInputMessage(role="user", content="你好")],
        model_name="qwen3-32b",
    )

    assert FakeChatModel.last_init_kwargs["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_llm_client_keeps_qwen3_stream_without_default_thinking_override(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证流式 Qwen3 调用不会默认注入 enable_thinking=false。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3-32b")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    async for _ in llm_client.stream_chat_completion(
        messages=[LlmInputMessage(role="user", content="你好")],
        model_name="qwen3-32b",
    ):
        pass

    assert FakeChatModel.last_init_kwargs["extra_body"] is None


@pytest.mark.asyncio
async def test_llm_client_streams_chunks(monkeypatch: MonkeyPatch) -> None:
    """验证 LLM 客户端会直接透传真实流式 chunk。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    streamed_chunks = [
        chunk
        async for chunk in llm_client.stream_chat_completion(
            messages=[],
            model_name="unit-test-model",
        )
    ]

    assert streamed_chunks == [
        LlmChatCompletionChunk(content_delta="模拟", model_name="unit-test-model"),
        LlmChatCompletionChunk(content_delta="流式回答", model_name="unit-test-model"),
        LlmChatCompletionChunk(
            model_name="unit-test-model",
            prompt_tokens=2,
            completion_tokens=4,
            total_tokens=6,
            finish_reason="stop",
        ),
    ]


@pytest.mark.asyncio
async def test_llm_client_logs_non_stream_request_messages(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证非流式调用前会打印完整的 LLM 输入消息。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)
    caplog.set_level(logging.INFO, logger="app.clients.llm_client")

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[
            LlmInputMessage(role="system", content="你是测试助手"),
            LlmInputMessage(role="user", content="请回答杭州在哪里"),
        ],
        model_name="unit-test-model",
    )

    assert "向 LLM 发起请求" in caplog.text
    assert '"mode": "non_stream"' in caplog.text
    assert "你是测试助手" in caplog.text
    assert "请回答杭州在哪里" in caplog.text


@pytest.mark.asyncio
async def test_llm_client_logs_stream_request_messages(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证流式调用前也会打印完整的 LLM 输入消息。"""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)
    caplog.set_level(logging.INFO, logger="app.clients.llm_client")

    llm_client = LlmClient()
    async for _ in llm_client.stream_chat_completion(
        messages=[LlmInputMessage(role="user", content="请流式回答这个问题")],
        model_name="unit-test-model",
    ):
        pass

    assert "向 LLM 发起请求" in caplog.text
    assert '"mode": "stream"' in caplog.text
    assert "请流式回答这个问题" in caplog.text


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
