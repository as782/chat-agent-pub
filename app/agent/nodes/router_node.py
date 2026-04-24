"""路由节点模块。
负责把当前请求映射到后续应执行的图分支。
当前阶段只做最小路由决策，不负责真正切换到知识库或 MCP 节点。
"""

from __future__ import annotations

from app.agent.router import resolve_agent_route
from app.agent.state import AgentState


class RouterNode:
    """LangGraph 路由节点。"""

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行路由判断。"""

        return {
            "route": resolve_agent_route(state),
            "forced_route": None,
        }
