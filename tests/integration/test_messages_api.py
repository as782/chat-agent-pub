"""消息接口集成测试。"""

from fastapi.testclient import TestClient


def test_list_messages_returns_message_history(app_client: TestClient) -> None:
    """验证消息历史接口可以返回同一会话下的消息。"""

    chat_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好，帮我记录这条消息"}],
        },
    )
    session_id = chat_response.headers["X-Session-ID"]

    response = app_client.get(f"/api/v1/messages/{session_id}")
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["total"] == 2
    assert [message["role"] for message in response_payload["items"]] == ["user", "assistant"]


def test_list_messages_returns_404_when_session_not_found(app_client: TestClient) -> None:
    """验证查询不存在会话的消息历史时返回 404。"""

    response = app_client.get("/api/v1/messages/not-exists")

    assert response.status_code == 404
    assert response.json()["error_code"] == "resource_not_found"
