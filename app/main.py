"""应用入口文件。

负责创建 FastAPI 应用并暴露最基础的健康检查接口。
当前阶段不负责业务路由编排、依赖注入容器和中间件装配。
"""

from fastapi import FastAPI


async def health_check() -> dict[str, str]:
    """返回服务健康状态，用于容器探针和本地联调。"""
    return {"status": "ok"}


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    application = FastAPI(
        title="Chat Agent Backend",
        version="0.1.0",
        description="最小可用的 Agent 问答系统后端。",
    )
    application.add_api_route("/health", health_check, methods=["GET"], tags=["system"])
    return application


app = create_app()
