"""Unit tests for the LLM client."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from openai import NotFoundError, PermissionDeniedError
from pytest import MonkeyPatch

from app.clients.llm_client import LlmClient, LlmInputMessage
from app.core.exceptions import ConfigurationException, UpstreamServiceException


class FakeChatModel:
    """Minimal fake chat model used by the tests."""

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
        """Record tool binding arguments and return itself."""

        self.__class__.last_tool_choice = tool_choice
        return self

    async def ainvoke(self, _: object) -> AIMessage:
        """Return a stable fake model response."""

        return AIMessage(content="mock model answer")

    async def astream(self, _: object) -> AsyncIterator[AIMessageChunk]:
        """Return a stable fake streaming response."""

        yield AIMessageChunk(content="mock", response_metadata={"model_name": "unit-test-model"})
        yield AIMessageChunk(
            content=" stream answer",
            response_metadata={"model_name": "unit-test-model"},
        )
        yield AIMessageChunk(
            content="",
            response_metadata={"model_name": "unit-test-model", "finish_reason": "stop"},
            usage_metadata={"input_tokens": 2, "output_tokens": 4, "total_tokens": 6},
        )


class FakePermissionDeniedModel(FakeChatModel):
    """Fake model used to simulate quota errors."""

    async def ainvoke(self, _: object) -> AIMessage:
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


class FakeNotFoundModel(FakeChatModel):
    """Fake model used to simulate missing model errors."""

    async def ainvoke(self, _: object) -> AIMessage:
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(status_code=404, request=request)
        raise NotFoundError(
            "model not found",
            response=response,
            body={
                "error": {
                    "message": (
                        "The model `Qwen3-32B` does not exist or you do not have access "
                        "to it."
                    ),
                    "code": "model_not_found",
                }
            },
        )


@pytest.mark.asyncio
async def test_llm_client_builds_chat_model_from_settings(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    answer = await llm_client.generate_answer("hello")

    assert answer == "mock model answer"
    assert FakeChatModel.last_init_kwargs["model"] == "unit-test-model"
    assert FakeChatModel.last_init_kwargs["model_provider"] == "openai"
    assert FakeChatModel.last_init_kwargs["api_key"] == "unit-test-key"
    assert FakeChatModel.last_init_kwargs["base_url"] == "https://example.com/v1"
    assert isinstance(FakeChatModel.last_init_kwargs["timeout"], httpx.Timeout)
    assert FakeChatModel.last_init_kwargs["timeout"].connect == 60.0
    assert FakeChatModel.last_init_kwargs["extra_body"] is None


@pytest.mark.asyncio
async def test_llm_client_allows_overriding_base_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[LlmInputMessage(role="user", content="hello")],
        model_name="planner-model",
        base_url="https://planner.example.com/v1",
    )

    assert FakeChatModel.last_init_kwargs["base_url"] == "https://planner.example.com/v1"


@pytest.mark.asyncio
async def test_llm_client_allows_overriding_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[LlmInputMessage(role="user", content="hello")],
        model_name="planner-model",
        api_key="planner-test-key",
    )

    assert FakeChatModel.last_init_kwargs["api_key"] == "planner-test-key"


@pytest.mark.asyncio
async def test_llm_client_allows_overriding_timeout(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[LlmInputMessage(role="user", content="hello")],
        model_name="planner-model",
        timeout_seconds=18,
    )

    assert isinstance(FakeChatModel.last_init_kwargs["timeout"], httpx.Timeout)
    assert FakeChatModel.last_init_kwargs["timeout"].connect == 18


@pytest.mark.asyncio
async def test_llm_client_disables_qwen3_thinking_for_non_stream_calls(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3-32b")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[LlmInputMessage(role="user", content="hello")],
        model_name="qwen3-32b",
    )

    assert FakeChatModel.last_init_kwargs["extra_body"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.asyncio
async def test_llm_client_uses_default_thinking_from_settings(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setenv("OPENAI_ENABLE_THINKING", "true")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[LlmInputMessage(role="user", content="hello")],
        model_name="unit-test-model",
    )

    assert FakeChatModel.last_init_kwargs["extra_body"] == {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


@pytest.mark.asyncio
async def test_llm_client_keeps_qwen3_stream_without_default_thinking_override(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3-32b")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)

    llm_client = LlmClient()
    async for _ in llm_client.stream_chat_completion(
        messages=[LlmInputMessage(role="user", content="hello")],
        model_name="qwen3-32b",
    ):
        pass

    assert FakeChatModel.last_init_kwargs["extra_body"] is None


@pytest.mark.asyncio
async def test_llm_client_streams_chunks(monkeypatch: MonkeyPatch) -> None:
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

    assert len(streamed_chunks) == 3
    assert streamed_chunks[0].content == "mock"
    assert streamed_chunks[1].content == " stream answer"
    assert streamed_chunks[2].response_metadata["finish_reason"] == "stop"
    assert streamed_chunks[2].usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 4,
        "total_tokens": 6,
    }


@pytest.mark.asyncio
async def test_llm_client_logs_non_stream_request_messages(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)
    caplog.set_level(logging.INFO, logger="app.clients.llm_client")

    llm_client = LlmClient()
    await llm_client.create_chat_completion(
        messages=[
            LlmInputMessage(role="system", content="you are a test assistant"),
            LlmInputMessage(role="user", content="where is hangzhou"),
        ],
        model_name="unit-test-model",
    )

    assert "LLM" in caplog.text
    assert '"mode": "non_stream"' in caplog.text
    assert "you are a test assistant" in caplog.text
    assert "where is hangzhou" in caplog.text


@pytest.mark.asyncio
async def test_llm_client_logs_stream_request_messages(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeChatModel)
    caplog.set_level(logging.INFO, logger="app.clients.llm_client")

    llm_client = LlmClient()
    async for _ in llm_client.stream_chat_completion(
        messages=[LlmInputMessage(role="user", content="stream this answer")],
        model_name="unit-test-model",
    ):
        pass

    assert "LLM" in caplog.text
    assert '"mode": "stream"' in caplog.text
    assert "stream this answer" in caplog.text


@pytest.mark.asyncio
async def test_llm_client_maps_quota_error_to_upstream_exception(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakePermissionDeniedModel)

    llm_client = LlmClient()

    with pytest.raises(UpstreamServiceException) as exception_info:
        await llm_client.generate_answer("hello")

    assert exception_info.value.error_code == "llm_quota_exceeded"
    assert exception_info.value.status_code == 503


@pytest.mark.asyncio
async def test_llm_client_maps_model_not_found_to_upstream_exception(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    monkeypatch.setattr("app.clients.llm_client.init_chat_model", FakeNotFoundModel)

    llm_client = LlmClient()

    with pytest.raises(UpstreamServiceException) as exception_info:
        await llm_client.generate_answer("hello")

    assert exception_info.value.error_code == "llm_model_not_found"
    assert exception_info.value.status_code == 404
    assert exception_info.value.details["provider_error_code"] == "model_not_found"


def test_llm_client_raises_configuration_error_when_api_key_missing(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    llm_client = LlmClient()

    with pytest.raises(ConfigurationException):
        llm_client._create_chat_model()
