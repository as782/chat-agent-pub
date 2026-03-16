"""对话接口集成测试。"""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.clients.llm_client import LlmChatCompletionChunk
from app.core.exceptions import UpstreamServiceException


def test_chat_api_creates_session_and_returns_answer(app_client: TestClient) -> None:
    """验证内部聊天接口会返回 OpenAI 兼容响应，并通过响应头暴露会话标识。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好，系统"}],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]

    assert response.status_code == 200
    assert session_id
    assert response_payload["id"].startswith("chatcmpl-")
    assert response_payload["object"] == "chat.completion"
    assert response_payload["model"] == "test-model"
    assert response_payload["choices"][0]["message"]["role"] == "assistant"
    assert response_payload["choices"][0]["message"]["content"] == "测试模型回答：你好，系统"
    assert response_payload["choices"][0]["finish_reason"] == "stop"
    assert response_payload["usage"]["total_tokens"] == 20


def test_chat_api_executes_builtin_tool_when_enabled(app_client: TestClient) -> None:
    """验证内部聊天接口会在后台执行工具，但对外仍返回 OpenAI 兼容格式。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "请帮我计算 1+1"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
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
    assert response_payload["choices"][0]["message"]["content"] == "测试模型回答：工具结果是 2"
    assert response_payload["choices"][0]["finish_reason"] == "stop"
    assert history_response.status_code == 200
    assert history_payload["total"] == 4
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history_payload["items"][1]["metadata"]["tool_calls"][0]["tool_name"] == "calculator"
    assert history_payload["items"][2]["content"] == "2"


def test_chat_api_supports_multi_turn_memory(app_client: TestClient) -> None:
    """验证同一会话下会自动注入历史消息，实现最小多轮记忆。"""

    first_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )
    session_id = first_response.headers["X-Session-ID"]

    second_response = app_client.post(
        "/api/v1/chat",
        headers={"X-Session-ID": session_id},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我刚刚告诉你的名字是什么？"}],
        },
    )

    assert second_response.status_code == 200
    assert (
        second_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )


def test_chat_api_combines_session_memory_and_explicit_messages(app_client: TestClient) -> None:
    """验证带 session_id 时，会结合系统历史和本次显式 messages 回答。"""

    first_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )
    session_id = first_response.headers["X-Session-ID"]

    second_response = app_client.post(
        "/api/v1/chat",
        headers={"X-Session-ID": session_id},
        json={
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": "请结合系统记录和本次输入回答。",
                },
                {"role": "user", "content": "我刚刚告诉你的名字是什么？"},
            ],
        },
    )

    assert second_response.status_code == 200
    assert (
        second_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )


def test_chat_api_does_not_use_other_session_memory_without_session_id(
    app_client: TestClient,
) -> None:
    """验证不携带 session_id 时，不会读取其他会话已保存的系统记忆。"""

    first_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )

    second_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "我叫小王，请记住这个名字。"},
                {"role": "assistant", "content": "好的，我记住了。"},
                {"role": "user", "content": "我刚刚告诉你的名字是什么？"},
            ],
        },
    )

    assert second_response.status_code == 200
    assert (
        second_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )
    assert second_response.headers["X-Session-ID"] != first_response.headers["X-Session-ID"]


def test_chat_api_streams_response_when_requested(app_client: TestClient) -> None:
    """验证内部聊天接口在 stream=true 时返回 OpenAI 兼容 SSE 数据。"""

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["X-Session-ID"]
    assert '"object": "chat.completion.chunk"' in response_body
    assert '"role": "assistant"' in response_body
    assert response_body.count('"content":') >= 2
    assert "[DONE]" in response_body


def test_chat_api_returns_404_when_session_not_found(app_client: TestClient) -> None:
    """验证对话接口在会话不存在时返回 404。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "继续回答"}],
        },
        headers={"X-Session-ID": "not-exists"},
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "resource_not_found"


def test_chat_api_stream_returns_json_error_when_first_chunk_fails(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证内部流式接口在首块失败时返回 JSON 错误，而不是直接中断连接。"""

    def fake_stream_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> AsyncIterator[LlmChatCompletionChunk]:
        """模拟首个流式块生成前发生限流错误。"""

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
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    )

    assert response.status_code == 429
    assert response.json()["error_code"] == "llm_rate_limited"
