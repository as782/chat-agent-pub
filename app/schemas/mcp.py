"""MCP 领域数据模型。
负责定义 MCP 服务列表、探测、工具列表和工具调用所需的数据结构。
当前阶段只覆盖最小可用能力，不负责完整 MCP 协议对象建模。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class McpServerInfo(BaseModel):
    """MCP 服务信息模型。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="MCP 服务名称。")
    transport: Literal["http", "streamable_http", "stdio"] = Field(description="连接方式。")
    endpoint: str | None = Field(default=None, description="HTTP 端点地址。")
    command: str | None = Field(default=None, description="stdio 启动命令。")
    is_enabled: bool = Field(default=True, description="当前是否启用。")


class McpServerListResponse(BaseModel):
    """MCP 服务列表响应。"""

    model_config = ConfigDict(extra="forbid")

    items: list[McpServerInfo] = Field(default_factory=list, description="服务列表。")


class McpProbeResponse(BaseModel):
    """MCP 服务探测结果。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="MCP 服务名称。")
    transport: Literal["http", "streamable_http", "stdio"] = Field(description="连接方式。")
    is_available: bool = Field(description="当前是否可用。")
    detail: str = Field(description="探测说明。")


class McpMethodCallRequest(BaseModel):
    """MCP 通用方法调用请求。"""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1, description="要调用的方法名。")
    params: dict[str, Any] = Field(default_factory=dict, description="方法参数。")


class McpMethodCallResponse(BaseModel):
    """MCP 通用方法调用响应。"""

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(description="目标服务名称。")
    method: str = Field(description="调用的方法名。")
    result: Any = Field(default=None, description="MCP 返回结果。")


class McpToolInfo(BaseModel):
    """MCP 工具信息模型。"""

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(description="所属服务名称。")
    name: str = Field(description="远端原始工具名。")
    registered_name: str = Field(description="绑定给模型时使用的注册名。")
    description: str | None = Field(default=None, description="工具描述。")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="工具入参 schema。")


class McpToolListResponse(BaseModel):
    """MCP 工具列表响应。"""

    model_config = ConfigDict(extra="forbid")

    items: list[McpToolInfo] = Field(default_factory=list, description="工具列表。")


class McpToolCallRequest(BaseModel):
    """MCP 工具调用请求。"""

    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数。")


class McpToolCallResponse(BaseModel):
    """MCP 工具调用响应。"""

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(description="目标服务名称。")
    tool_name: str = Field(description="调用的工具名。")
    arguments: dict[str, Any] = Field(default_factory=dict, description="本次调用参数。")
    content: list[dict[str, Any]] = Field(default_factory=list, description="原始内容块。")
    structured_content: dict[str, Any] | None = Field(
        default=None,
        description="结构化结果。",
    )
    is_error: bool = Field(default=False, description="远端是否标记为错误结果。")
    output_text: str = Field(default="", description="归一化文本输出。")
