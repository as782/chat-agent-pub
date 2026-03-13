"""健康检查接口集成测试。"""

from fastapi.testclient import TestClient


def test_health_check_returns_ok(app_client: TestClient) -> None:
    """验证健康检查接口返回成功状态。"""

    response = app_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_check_with_invalid_method_returns_405(app_client: TestClient) -> None:
    """验证健康检查接口对不支持的方法返回 405。"""

    response = app_client.post("/health")

    assert response.status_code == 405
