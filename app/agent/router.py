"""Agent 路由决策模块。
负责根据当前请求判断对话图应优先走哪条逻辑分支。
当前阶段仍然使用显式规则路由，不负责 LLM Planner 式决策。
"""

from __future__ import annotations

from app.agent.state import AgentRoute, AgentState


def resolve_agent_route(state: AgentState) -> AgentRoute:
    """根据当前状态推导目标路由。"""

    latest_user_message = str(state.get("latest_user_message", ""))
    normalized_message = latest_user_message.lower()
    forced_route = state.get("forced_route")
    scheduled_route = state.get("scheduled_route")

    if forced_route is not None:
        return forced_route
    if scheduled_route is not None:
        return scheduled_route
    if state.get("requested_tool_names"):
        return "tool"
    execution_plan = state.get("execution_plan")
    if execution_plan is not None:
        return execution_plan.recommended_route
    if _is_knowledge_request(
        raw_message=latest_user_message,
        normalized_message=normalized_message,
    ):
        return "ragflow"
    if latest_user_message.startswith("service:") or "#service" in normalized_message:
        return "service"
    if latest_user_message.startswith("route:") or "#route" in normalized_message:
        return "route"
    if latest_user_message.startswith("mcp:") or "#mcp" in normalized_message:
        return "mcp"
    return "answer"


def _is_knowledge_request(*, raw_message: str, normalized_message: str) -> bool:
    """判断当前问题是否显式要求走知识库检索。"""

    return (
        raw_message.startswith("知识库:")
        or normalized_message.startswith("knowledge:")
        or normalized_message.startswith("konwledge:")
        or "#knowledge" in normalized_message
    )
