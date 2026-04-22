"""规划节点模块。
负责在正式路由前输出当前问题的主分类和最小执行计划。
当前阶段先使用规则式规划器，不直接改变既有图分支行为。
"""

from __future__ import annotations

from app.agent.planner import PlannerService
from app.agent.state import AgentState
from app.clients.llm_client import LlmClient


class PlannerNode:
    """LangGraph 规划节点。"""

    def __init__(
        self,
        *,
        planner_service: PlannerService | None = None,
        llm_client: LlmClient | None = None,
    ) -> None:
        self._planner_service = planner_service or PlannerService(llm_client=llm_client)

    async def run(self, state: AgentState) -> dict[str, object]:
        """生成主分类和最小执行计划。"""

        execution_plan = await self._planner_service.build_plan_async(state)
        return {
            "primary_category": execution_plan.primary_category,
            "execution_plan": execution_plan,
            "need_clarification": execution_plan.need_clarification,
            "clarification_question": execution_plan.clarification_question,
            "steps": execution_plan.steps,
            
        }
