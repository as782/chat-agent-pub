"""对话接口集成测试。"""

from fastapi.testclient import TestClient


def test_chat_api_creates_session_and_returns_answer(app_client: TestClient) -> None:
    """验证对话接口在未提供会话时会创建会话并返回回答。"""

    response = app_client.post(
        "/api/v1/chat",
        json={"user_message": "你好，系统"},
    )
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["session_id"]
    assert response_payload["used_knowledge"] is False
    assert response_payload["used_tools"] == []
    assert response_payload["answer"] == "测试模型回答：你好，系统"
    assert response_payload["finish_reason"] == "stop"
    assert response_payload["model"] == "test-model"


def test_chat_api_executes_builtin_tool_when_enabled(app_client: TestClient) -> None:
    """验证内部聊天接口会在启用工具时执行内置工具。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "user_message": "请帮我计算 1+1",
            "enable_tools": True,
        },
    )
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["used_tools"] == ["calculator"]
    assert response_payload["tool_calls"][0]["tool_name"] == "calculator"
    assert response_payload["tool_calls"][0]["output"] == "2"
    assert response_payload["answer"] == "测试模型回答：工具结果是 2"


def test_chat_api_streams_response_when_requested(app_client: TestClient) -> None:
    """验证内部聊天接口在 stream=true 时返回 SSE 数据。"""

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={"user_message": "你好", "stream": True},
    ) as response:
        response_body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"type": "message_start"' in response_body
    assert '"type": "answer_delta"' in response_body
    assert '"type": "message_end"' in response_body


def test_chat_api_returns_404_when_session_not_found(app_client: TestClient) -> None:
    """验证对话接口在会话不存在时返回 404。"""

    response = app_client.post(
        "/api/v1/chat",
        json={"session_id": "not-exists", "user_message": "继续回答"},
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "resource_not_found"
