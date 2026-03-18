"""Agent 路由决策模块。
负责根据当前请求判断对话图应优先走哪条逻辑分支。
当前阶段仍然使用显式规则路由，不负责 LLM Planner 式决策。
"""

from __future__ import annotations

import logging

from app.agent.state import AgentRoute, AgentState

LOGGER = logging.getLogger(__name__)


def resolve_agent_route(state: AgentState) -> AgentRoute:
    """根据当前状态推导目标路由。"""

    latest_user_message = str(state.get("latest_user_message", ""))[:200]
    normalized_message = latest_user_message.lower()
    scheduled_route = state.get("scheduled_route")

    LOGGER.info(
        (
            "========== 路由决策开始 ==========\n"
            "用户问题：%s\n"
            "scheduled_route: %s\n"
            "requested_tool_names: %s\n"
            "execution_plan.recommended_route: %s"
        ),
        latest_user_message,
        scheduled_route,
        state.get("requested_tool_names"),
        state.get("execution_plan").recommended_route if state.get("execution_plan") else None,
    )

    if scheduled_route is not None:
        LOGGER.info("最终路由：%s（使用 scheduled_route）", scheduled_route)
        return scheduled_route
    
    if state.get("requested_tool_names"):
        LOGGER.info("最终路由：tool（用户显式请求工具）")
        return "tool"
    
    execution_plan = state.get("execution_plan")
    if execution_plan is not None:
        route = execution_plan.recommended_route
        LOGGER.info(
            "最终路由：%s（来自 execution_plan，primary_category=%s）",
            route,
            execution_plan.primary_category,
        )
        return route
    
    # 后备规则判断
    if _is_knowledge_request(
        raw_message=latest_user_message,
        normalized_message=normalized_message,
    ):
        LOGGER.info("最终路由：ragflow（显式知识库请求）")
        return "ragflow"
    
    if latest_user_message.startswith("route:") or "#route" in normalized_message:
        LOGGER.info("最终路由：route（显式路线请求）")
        return "route"
    
    if latest_user_message.startswith("mcp:") or "#mcp" in normalized_message:
        LOGGER.info("最终路由：mcp（显式 MCP 请求）")
        return "mcp"
    
    LOGGER.info("最终路由：answer（默认）")
    return "answer"


def _is_knowledge_request(*, raw_message: str, normalized_message: str) -> bool:
    """判断当前问题是否显式要求走知识库检索。"""

    return (
        raw_message.startswith("知识库:")
        or normalized_message.startswith("knowledge:")
        or normalized_message.startswith("konwledge:")
        or "#knowledge" in normalized_message
    )
