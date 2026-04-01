"""异常模块。

负责定义统一的业务异常，并向 FastAPI 注册统一异常处理器。
当前阶段不负责复杂错误码治理、国际化文案和外部告警联动。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.logger import get_logger

LOGGER = get_logger(__name__)


def _resolve_root_cause(exception: Exception) -> Exception:
    """沿异常链向下查找最内层根因。"""

    visited_exception_ids: set[int] = set()
    current_exception: Exception = exception
    while True:
        if id(current_exception) in visited_exception_ids:
            return current_exception
        visited_exception_ids.add(id(current_exception))

        next_exception = current_exception.__cause__ or current_exception.__context__
        if not isinstance(next_exception, Exception):
            return current_exception
        current_exception = next_exception


class AppException(Exception):
    """应用通用异常。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "app_error",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ResourceNotFoundException(AppException):
    """资源不存在异常。"""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            error_code="resource_not_found",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class ConfigurationException(AppException):
    """配置缺失或配置错误异常。"""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            error_code="configuration_error",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class UpstreamServiceException(AppException):
    """上游服务调用异常。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "upstream_service_error",
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


async def app_exception_handler(_: Request, exception: AppException) -> JSONResponse:
    """处理自定义应用异常并返回统一错误响应。"""

    return JSONResponse(
        status_code=exception.status_code,
        content={
            "error_code": exception.error_code,
            "message": exception.message,
            "details": exception.details,
        },
    )


async def http_exception_handler(_: Request, exception: HTTPException) -> JSONResponse:
    """处理 FastAPI 标准异常并对外暴露统一结构。"""

    return JSONResponse(
        status_code=exception.status_code,
        content={
            "error_code": "http_error",
            "message": str(exception.detail),
            "details": {},
        },
    )


async def unhandled_exception_handler(request: Request, exception: Exception) -> JSONResponse:
    """处理未捕获异常，避免对外暴露内部堆栈。"""

    LOGGER.exception("发生未处理异常。", exc_info=exception)
    root_cause = _resolve_root_cause(exception)
    resolved_message = str(root_cause).strip() or "服务内部异常"
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "internal_server_error",
            "message": "服务内部异常",
            "details": {
                "message": resolved_message,
            },
        },
    )


def register_exception_handlers(application: FastAPI) -> None:
    """向 FastAPI 应用注册统一异常处理器。"""

    application.add_exception_handler(AppException, app_exception_handler)
    application.add_exception_handler(HTTPException, http_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)
