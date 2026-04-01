"""异常模块单元测试。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import AppException, register_exception_handlers


def test_app_exception_handler_returns_unified_payload() -> None:
    """验证自定义异常会返回统一错误结构。"""

    application = FastAPI()
    register_exception_handlers(application)

    @application.get("/app-error")
    async def raise_app_exception() -> None:
        """抛出应用异常。"""

        raise AppException(
            "参数错误",
            error_code="invalid_request",
            details={"field": "session_id"},
        )

    client = TestClient(application)
    response = client.get("/app-error")

    assert response.status_code == 400
    assert response.json() == {
        "error_code": "invalid_request",
        "message": "参数错误",
        "details": {"field": "session_id"},
    }


def test_unhandled_exception_handler_returns_internal_error() -> None:
    """验证未处理异常会被统一转换为 500。"""

    application = FastAPI()
    register_exception_handlers(application)

    @application.get("/unexpected-error")
    async def raise_unhandled_exception() -> None:
        """抛出未处理异常。"""

        raise RuntimeError("boom")

    client = TestClient(application, raise_server_exceptions=False)
    response = client.get("/unexpected-error")

    assert response.status_code == 500
    assert response.json() == {
        "error_code": "internal_server_error",
        "message": "服务内部异常",
        "details": {"message": "boom"},
    }


def test_unhandled_exception_handler_includes_root_cause_details() -> None:
    """验证未处理异常会把异常链中的根因消息透出到 message。"""

    application = FastAPI()
    register_exception_handlers(application)

    @application.get("/wrapped-error")
    async def raise_wrapped_exception() -> None:
        """抛出带根因的包装异常。"""

        try:
            raise ValueError("model_not_found")
        except ValueError as exception:
            raise RuntimeError("graph failed") from exception

    client = TestClient(application, raise_server_exceptions=False)
    response = client.get("/wrapped-error")
    response_payload = response.json()

    assert response.status_code == 500
    assert response_payload == {
        "error_code": "internal_server_error",
        "message": "服务内部异常",
        "details": {"message": "model_not_found"},
    }
