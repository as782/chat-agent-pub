"""MCP 接口模块。

负责暴露 MCP 服务器列表、探测和通用方法调用接口。
当前阶段不负责复杂权限控制和完整会话级代理。
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.mcp.manager import McpManager
from app.schemas.mcp import (
    McpMethodCallRequest,
    McpMethodCallResponse,
    McpProbeResponse,
    McpServerListResponse,
)

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/servers", response_model=McpServerListResponse, status_code=status.HTTP_200_OK)
async def list_mcp_servers() -> McpServerListResponse:
    """查询当前已配置的 MCP 服务器列表。"""

    mcp_manager = McpManager()
    return McpServerListResponse(items=mcp_manager.list_servers())


@router.post(
    "/servers/{server_name}/probe",
    response_model=McpProbeResponse,
    status_code=status.HTTP_200_OK,
)
async def probe_mcp_server(server_name: str) -> McpProbeResponse:
    """探测指定 MCP 服务器当前是否可用。"""

    mcp_manager = McpManager()
    return await mcp_manager.probe_server(server_name)


@router.post(
    "/servers/{server_name}/call",
    response_model=McpMethodCallResponse,
    status_code=status.HTTP_200_OK,
)
async def call_mcp_server_method(
    server_name: str,
    request: McpMethodCallRequest,
) -> McpMethodCallResponse:
    """调用指定 MCP 服务器的一个方法。"""

    mcp_manager = McpManager()
    result = await mcp_manager.call_server_method(
        server_name=server_name,
        method=request.method,
        params=request.params,
    )
    return McpMethodCallResponse(
        server_name=server_name,
        method=request.method,
        result=result,
    )
