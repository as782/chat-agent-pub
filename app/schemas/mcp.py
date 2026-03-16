"""MCP 领域数据模型。

负责定义 MCP 服务器列表、探测和通用方法调用所需的数据结构。
当前阶段只覆盖最小骨架能力，不负责完整 MCP 协议对象建模。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class McpServerInfo(BaseModel):
    """MCP 服务器信息模型。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="MCP 服务器名称。")
    transport: Literal["http", "stdio"] = Field(description="连接方式。")
    endpoint: str | None = Field(default=None, description="HTTP 端点地址。")
    command: str | None = Field(default=None, description="stdio 启动命令。")
    is_enabled: bool = Field(default=True, description="当前是否启用。")


class McpServerListResponse(BaseModel):
    """MCP 服务器列表响应。"""

    model_config = ConfigDict(extra="forbid")

    items: list[McpServerInfo] = Field(default_factory=list, description="服务器列表。")


class McpProbeResponse(BaseModel):
    """MCP 服务器探测结果。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="MCP 服务器名称。")
    transport: Literal["http", "stdio"] = Field(description="连接方式。")
    is_available: bool = Field(description="当前是否可用。")
    detail: str = Field(description="探测说明。")


class McpMethodCallRequest(BaseModel):
    """MCP 方法调用请求。"""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1, description="要调用的方法名。")
    params: dict[str, Any] = Field(default_factory=dict, description="方法参数。")


class McpMethodCallResponse(BaseModel):
    """MCP 方法调用响应。"""

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(description="目标服务器名称。")
    method: str = Field(description="调用的方法名。")
    result: Any = Field(default=None, description="MCP 返回结果。")
