"""OpenAI 兼容接口集成测试。"""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.core.exceptions import UpstreamServiceException


def test_openai_compat_chat_completions_returns_standard_response(
    app_client: TestClient,
) -> None:
    """验证兼容接口复用内部聊天主链路，并暴露会话标识。"""

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
    session_id = response.headers["X-Session-ID"]

    assert response.status_code == 200
    assert session_id
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
    """验证兼容接口在传入 tools 时会执行工具并返回最终回答。"""

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
    session_id = response.headers["X-Session-ID"]

    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert response_payload["choices"][0]["finish_reason"] == "stop"
    assert response_payload["choices"][0]["message"]["content"] == "测试模型回答：工具结果是 2"
    assert history_response.status_code == 200
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history_payload["items"][2]["content"] == "2"


def test_openai_compat_chat_completions_supports_multi_turn_memory(
    app_client: TestClient,
) -> None:
    """验证兼容接口支持通过 X-Session-ID 复用内部会话记忆。"""

    first_response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )
    session_id = first_response.headers["X-Session-ID"]

    second_response = app_client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "我刚刚告诉你的名字是什么？"}],
        },
    )

    assert second_response.status_code == 200
    assert (
        second_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )


def test_openai_compat_chat_completions_streams_response(app_client: TestClient) -> None:
    """验证兼容接口在 stream=true 时返回与内部聊天一致的 SSE。"""

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
    assert response.headers["X-Session-ID"]
    assert '"object": "chat.completion.chunk"' in response_body
    assert response_body.count('"role": "assistant"') >= 2
    assert response_body.count('"content":') >= 2
    assert "[DONE]" in response_body


def test_openai_compat_chat_completions_wraps_reasoning_content_in_content(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """Reasoning content should be wrapped into content on non-stream OpenAI-compatible responses."""

    from app.agent.state import ChatTurnResult

    async def fake_send_message(self, request, session_id=None):
        del request, session_id
        return (
            "session-001",
            self._openai_compat_service.build_chat_completion_response(
                ChatTurnResult(
                    session_id="session-001",
                    content="最终回答",
                    model_name="qwen-compatible-model",
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                    finish_reason="stop",
                    route="answer",
                    reasoning_content="这是模型的思考过程",
                )
            ),
        )

    monkeypatch.setattr("app.services.chat_service.ChatService.send_message", fake_send_message)

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "???"}],
        },
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == "<think>???????????????</think>???????"
    )
    assert "reasoning_content" not in response.json()["choices"][0]["message"]


def test_openai_compat_chat_completions_streams_wrapped_reasoning_content(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """Reasoning content should be wrapped into streaming OpenAI-compatible deltas."""

    def fake_stream_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[AIMessageChunk]:
        del self, messages, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking

        async def iterator() -> AsyncIterator[AIMessageChunk]:
            yield AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "?????????"},
                response_metadata={"model_name": model_name or "test-model"},
            )
            yield AIMessageChunk(
                content="???????",
                response_metadata={"model_name": model_name or "test-model"},
            )
            yield AIMessageChunk(
                content="",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

        return iterator()

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.stream_chat_completion",
        fake_stream_chat_completion,
    )

    with app_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "???"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert '"content": "<think>?????????' in response_body
    assert '"content": "</think>???????' in response_body
    assert '"reasoning_content"' not in response_body
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
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[AIMessageChunk]:
        """模拟在第一个流式块之前就发生上游限流错误。"""

        del self, messages, model_name, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking

        async def iterator() -> AsyncIterator[AIMessageChunk]:
            raise UpstreamServiceException(
                "LLM 提供方触发限流，请稍后重试。",
                error_code="llm_rate_limited",
                status_code=429,
            )
            yield AIMessageChunk(content="")

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
