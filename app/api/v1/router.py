"""API 路由聚合模块。

负责汇总 v1 版本下已开放的接口路由。
当前阶段已接入知识库与 MCP 路由。
"""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.knowledge import router as knowledge_router
from app.api.v1.mcp import router as mcp_router
from app.api.v1.messages import router as messages_router
from app.api.v1.sessions import router as sessions_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(sessions_router)
api_router.include_router(messages_router)
api_router.include_router(chat_router)
api_router.include_router(knowledge_router)
api_router.include_router(mcp_router)
