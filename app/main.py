"""应用入口文件。

负责创建 FastAPI 应用并暴露最基础的健康检查接口。
当前阶段不负责业务路由编排、依赖注入容器和中间件装配。
"""

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


async def health_check() -> dict[str, str]:
    """返回服务健康状态，用于容器探针和本地联调。"""

    return {"status": "ok"}


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""

    settings = get_settings()
    configure_logging(is_debug=settings.is_debug)

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="最小可用的 Agent 问答系统后端。",
    )
    register_exception_handlers(application)
    application.add_api_route("/health", health_check, methods=["GET"], tags=["system"])
    LOGGER.info("FastAPI 应用创建完成。")
    return application


app = create_app()
