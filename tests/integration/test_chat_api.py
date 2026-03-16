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


def test_chat_api_returns_404_when_session_not_found(app_client: TestClient) -> None:
    """验证对话接口在会话不存在时返回 404。"""

    response = app_client.post(
        "/api/v1/chat",
        json={"session_id": "not-exists", "user_message": "继续回答"},
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "resource_not_found"
