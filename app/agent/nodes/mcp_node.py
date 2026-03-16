"""MCP 节点模块。

负责在命中 MCP 路由时整理可用 MCP 服务信息，并注入回答上下文。
当前阶段只做服务说明注入，不自动执行远端 MCP tool。
"""

from __future__ import annotations

from app.agent.state import AgentState
from app.mcp.manager import McpManager


class McpNode:
    """LangGraph MCP 节点。"""

    def __init__(self, *, mcp_manager: McpManager | None = None) -> None:
        self._mcp_manager = mcp_manager or McpManager()

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行 MCP 节点主逻辑。"""

        del state
        return {
            "mcp_context": self._mcp_manager.build_agent_context(),
        }
