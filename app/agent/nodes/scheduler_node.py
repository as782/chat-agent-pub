"""调度节点模块。
负责在 planner 和 argument 之后，根据 execution_plan 与已完成步骤选出当前轮次的执行分支。
当前阶段先做顺序调度，不直接展开并行执行。
"""

from __future__ import annotations

from app.agent.scheduler import SchedulerService
from app.agent.state import AgentState
from app.core.logger import get_logger

LOGGER = get_logger(__name__)


class SchedulerNode:
    """LangGraph 调度节点。"""

    def __init__(self, *, scheduler_service: SchedulerService | None = None) -> None:
        self._scheduler_service = scheduler_service or SchedulerService()

    async def run(self, state: AgentState) -> dict[str, object]:
        """输出当前轮次的调度结果。"""

        execution_plan = state.get("execution_plan")
        if execution_plan:
            LOGGER.info(
                "========== 任务调度开始 ==========\n"
                "执行模式：%s\n"
                "总步骤数：%s",
                execution_plan.execution_mode,
                len(execution_plan.steps),
            )
            for idx, step in enumerate(execution_plan.steps, 1):
                LOGGER.info("  步骤 %s: [%s] %s (depends_on=%s)", 
                           idx, step.step_id, step.goal, step.depends_on)

        scheduled_execution = self._scheduler_service.schedule(state)
        
        LOGGER.info(
            "调度结果：\n"
            "  scheduled_route: %s\n"
            "  current_step_id: %s\n"
            "  runnable_steps: %s\n"
            "  completed_steps: %s\n"
            "  pending_steps: %s",
            scheduled_execution.route,
            scheduled_execution.current_step_id,
            scheduled_execution.runnable_step_ids,
            scheduled_execution.completed_step_ids,
            scheduled_execution.pending_step_ids,
        )
        LOGGER.info("================================\n")
        
        return {
            "scheduled_route": scheduled_execution.route,
            "current_step_id": scheduled_execution.current_step_id,
            "runnable_step_ids": scheduled_execution.runnable_step_ids,
            "completed_step_ids": scheduled_execution.completed_step_ids,
            "pending_step_ids": scheduled_execution.pending_step_ids,
        }
