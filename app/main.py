"""应用入口文件。

负责创建 FastAPI 应用并暴露基础系统路由与业务路由。
当前阶段不负责复杂中间件装配和完整生产环境运维编排。
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.openai_compat import router as openai_compat_router
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import configure_logging, get_logger
from app.persistence.database import dispose_database, initialize_database

LOGGER = get_logger(__name__)


async def health_check() -> dict[str, str]:
    """返回服务健康状态，用于容器探针和本地联调。"""

    return {"status": "ok"}


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""

    settings = get_settings()
    configure_logging(settings=settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """在应用生命周期中初始化和释放基础资源。"""

        await initialize_database()
        proxy_http_client: httpx.AsyncClient | None = None
        if settings.enable_monitor_network_proxy:
            from live_agent_proxy.main import build_proxy_http_client

            proxy_http_client = build_proxy_http_client()
            application.state.monitor_network_proxy_http_client = proxy_http_client
        yield
        if proxy_http_client is not None:
            await proxy_http_client.aclose()
        await dispose_database()

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="最小可用的 Agent 问答系统后端。",
        lifespan=lifespan,
    )
    register_exception_handlers(application)
    application.include_router(openai_compat_router)
    application.include_router(api_router)
    if settings.enable_monitor_network_proxy:
        from live_agent_proxy.main import proxy_router

        application.include_router(proxy_router)
        LOGGER.info("监控网接口代理服务已启用。")
    application.add_api_route("/health", health_check, methods=["GET"], tags=["system"])
    LOGGER.info("FastAPI 应用创建完成。")
    return application


app = create_app()
