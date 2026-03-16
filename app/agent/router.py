"""Agent 路由决策模块。
负责根据当前请求判断对话图应优先走哪条逻辑分支。
当前阶段只输出最小可用决策结果，不负责真正切换到知识库或 MCP 节点。
"""

from __future__ import annotations

from app.agent.state import AgentRoute, AgentState


def resolve_agent_route(state: AgentState) -> AgentRoute:
    """根据当前状态推导目标路由。

    这里先把未来会拆分出去的分支显式标记出来，
    这样后续阶段只需要补节点与边，不需要重写整个判定入口。
    """

    latest_user_message = state.get("latest_user_message", "")
    normalized_message = latest_user_message.lower()

    if state.get("requested_tool_names"):
        return "tool"
    if latest_user_message.startswith("知识库:") or "#knowledge" in normalized_message:
        return "ragflow"
    if latest_user_message.startswith("mcp:") or "#mcp" in normalized_message:
        return "mcp"
    return "answer"
