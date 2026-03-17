"""MCP 内部模型模块。
负责定义 MCP 客户端、管理器和 Agent 节点之间共享的内部数据结构。
当前阶段不负责对外 API 响应建模，对外响应仍由 schemas 层负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

McpTransport = Literal["http", "streamable_http", "sse", "stdio"]


@dataclass(slots=True)
class McpClientToolDefinition:
    """MCP 远端工具定义。"""

    name: str
    description: str | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None


@dataclass(slots=True)
class McpClientToolCallResult:
    """MCP 远端工具调用结果。"""

    content: list[dict[str, Any]] = field(default_factory=list)
    structured_content: dict[str, Any] | None = None
    is_error: bool = False
    output_text: str = ""


@dataclass(slots=True)
class McpRuntimeTool:
    """Agent 在单次请求中可直接绑定给模型的 MCP 工具描述。"""

    registered_name: str
    server_name: str
    remote_tool_name: str
    description: str | None
    input_schema: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI 兼容 tool schema。"""

        return {
            "type": "function",
            "function": {
                "name": self.registered_name,
                "description": self.description or "",
                "parameters": self.input_schema
                if self.input_schema
                else {"type": "object", "properties": {}},
            },
        }
