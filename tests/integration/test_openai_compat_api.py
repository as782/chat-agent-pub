"""OpenAI 兼容接口集成测试。"""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.clients.llm_client import LlmChatCompletionChunk
from app.core.exceptions import UpstreamServiceException


def test_openai_compat_chat_completions_returns_standard_response(
    app_client: TestClient,
) -> None:
    """验证兼容接口会返回 OpenAI Chat Completions 兼容结构。"""

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [
                {"role": "system", "content": "你是一个简洁助手。"},
                {"role": "user", "content": "你好"},
            ],
            "user": "demo-user",
        },
    )
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["id"].startswith("chatcmpl-")
    assert response_payload["object"] == "chat.completion"
    assert response_payload["model"] == "qwen-compatible-model"
    assert response_payload["choices"][0]["message"]["role"] == "assistant"
    assert response_payload["choices"][0]["message"]["content"] == "测试模型回答：你好"
    assert response_payload["choices"][0]["finish_reason"] == "stop"
    assert response_payload["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }


def test_openai_compat_chat_completions_returns_tool_calls(app_client: TestClient) -> None:
    """验证兼容接口在传入 tools 时会返回 OpenAI 兼容 tool_calls。"""

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "请帮我计算 1+1"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "计算数学表达式。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                            },
                            "required": ["expression"],
                        },
                    },
                }
            ],
        },
    )
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["choices"][0]["finish_reason"] == "tool_calls"
    assert response_payload["choices"][0]["message"]["content"] is None
    assert (
        response_payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "calculator"
    )


def test_openai_compat_chat_completions_streams_response(app_client: TestClient) -> None:
    """验证兼容接口在 stream=true 时返回 OpenAI 兼容 SSE。"""

    with app_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"object": "chat.completion.chunk"' in response_body
    assert response_body.count('"content":') >= 2
    assert "[DONE]" in response_body


def test_openai_compat_chat_completions_rejects_unsupported_tool(app_client: TestClient) -> None:
    """验证兼容接口会拒绝未注册的工具名称。"""

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "你好"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "unknown_tool",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "unsupported_tool"


def test_openai_compat_stream_returns_json_error_when_first_chunk_fails(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证兼容流式接口在首块失败时返回正常 JSON 错误，而不是直接中断连接。"""

    def fake_stream_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> AsyncIterator[LlmChatCompletionChunk]:
        """模拟在第一个流式块之前就发生上游限流错误。"""

        del self, messages, model_name, tools, tool_choice

        async def iterator() -> AsyncIterator[LlmChatCompletionChunk]:
            raise UpstreamServiceException(
                "LLM 提供方触发限流，请稍后重试。",
                error_code="llm_rate_limited",
                status_code=429,
            )
            yield LlmChatCompletionChunk()

        return iterator()

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.stream_chat_completion",
        fake_stream_chat_completion,
    )

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    )

    assert response.status_code == 429
    assert response.json()["error_code"] == "llm_rate_limited"
