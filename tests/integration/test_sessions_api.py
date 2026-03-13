"""会话接口集成测试。"""

from fastapi.testclient import TestClient


def test_create_session_returns_created_session(app_client: TestClient) -> None:
    """验证创建会话接口返回新建会话。"""

    response = app_client.post(
        "/api/v1/sessions",
        json={"title": "新的会话", "user_id": "user-001"},
    )

    response_payload = response.json()

    assert response.status_code == 201
    assert response_payload["title"] == "新的会话"
    assert response_payload["user_id"] == "user-001"
    assert response_payload["status"] == "active"
    assert response_payload["session_id"]


def test_get_session_returns_404_when_session_not_found(app_client: TestClient) -> None:
    """验证查询不存在会话时返回 404。"""

    response = app_client.get("/api/v1/sessions/not-exists")

    assert response.status_code == 404
    assert response.json()["error_code"] == "resource_not_found"
