"""健康检查接口集成测试。"""

from fastapi.testclient import TestClient

from app.main import app


def test_health_check_returns_ok() -> None:
    """验证健康检查接口返回成功状态。"""
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
