"""MCP 接口模块。
负责暴露 MCP 服务列表、探测、工具发现和工具调用接口。
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
    McpToolCallRequest,
    McpToolCallResponse,
    McpToolListResponse,
)

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/servers", response_model=McpServerListResponse, status_code=status.HTTP_200_OK)
async def list_mcp_servers() -> McpServerListResponse:
    """查询当前已配置的 MCP 服务列表。"""

    mcp_manager = McpManager()
    return McpServerListResponse(items=mcp_manager.list_servers())


@router.post(
    "/servers/{server_name}/probe",
    response_model=McpProbeResponse,
    status_code=status.HTTP_200_OK,
)
async def probe_mcp_server(server_name: str) -> McpProbeResponse:
    """探测指定 MCP 服务当前是否可用。"""

    mcp_manager = McpManager()
    return await mcp_manager.probe_server(server_name)


@router.get(
    "/servers/{server_name}/tools",
    response_model=McpToolListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_mcp_server_tools(server_name: str) -> McpToolListResponse:
    """列出指定 MCP 服务的工具。"""

    mcp_manager = McpManager()
    return McpToolListResponse(items=await mcp_manager.list_tools(server_name))


@router.post(
    "/servers/{server_name}/tools/{tool_name}/call",
    response_model=McpToolCallResponse,
    status_code=status.HTTP_200_OK,
)
async def call_mcp_tool(
    server_name: str,
    tool_name: str,
    request: McpToolCallRequest,
) -> McpToolCallResponse:
    """调用指定 MCP 服务中的工具。"""

    mcp_manager = McpManager()
    return await mcp_manager.call_tool(
        server_name=server_name,
        tool_name=tool_name,
        arguments=request.arguments,
    )


@router.post(
    "/servers/{server_name}/call",
    response_model=McpMethodCallResponse,
    status_code=status.HTTP_200_OK,
)
async def call_mcp_server_method(
    server_name: str,
    request: McpMethodCallRequest,
) -> McpMethodCallResponse:
    """兼容旧接口的通用 MCP 方法调用。"""

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
