"""Agent 调度器模块。
负责根据 execution_plan、依赖关系和已完成步骤，决定当前轮次应执行哪个业务分支。
当前阶段先提供顺序调度骨架，并为后续并行 executor 预留 runnable_step_ids 输出。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.state import AgentRoute, AgentState, ExecutionPlan, ExecutionStep

EXECUTOR_ROUTE_MAPPING: dict[str, AgentRoute] = {
    "answer": "answer",
    "rag": "ragflow",
    "mcp": "mcp",
    "tool": "tool",
    "route": "route",
    "traffic": "traffic",
    "service": "service",
    "report": "report",
}


@dataclass(slots=True)
class ScheduledExecution:
    """当前轮次的调度结果。"""

    route: AgentRoute
    current_step_id: str | None
    runnable_step_ids: list[str]
    completed_step_ids: list[str]
    pending_step_ids: list[str]


class SchedulerService:
    """最小可用执行计划调度器。"""

    def schedule(self, state: AgentState) -> ScheduledExecution:
        """根据当前状态选出当前轮次应执行的步骤与路由。"""

        execution_plan = state.get("execution_plan")
        if execution_plan is None or not execution_plan.steps:
            return ScheduledExecution(
                route="answer",
                current_step_id=None,
                runnable_step_ids=[],
                completed_step_ids=[],
                pending_step_ids=[],
            )

        completed_step_ids = self._collect_completed_step_ids(execution_plan, state)
        pending_steps = [
            step for step in execution_plan.steps if step.step_id not in completed_step_ids
        ]
        runnable_steps = [
            step
            for step in pending_steps
            if all(dependency in completed_step_ids for dependency in step.depends_on)
        ]

        if not runnable_steps:
            return ScheduledExecution(
                route="answer",
                current_step_id=None,
                runnable_step_ids=[],
                completed_step_ids=completed_step_ids,
                pending_step_ids=[step.step_id for step in pending_steps],
            )

        current_step = self._select_current_step(execution_plan, runnable_steps)
        return ScheduledExecution(
            route=EXECUTOR_ROUTE_MAPPING.get(current_step.executor, "answer"),
            current_step_id=current_step.step_id,
            runnable_step_ids=[step.step_id for step in runnable_steps],
            completed_step_ids=completed_step_ids,
            pending_step_ids=[step.step_id for step in pending_steps],
        )

    @staticmethod
    def _collect_completed_step_ids(
        execution_plan: ExecutionPlan,
        state: AgentState,
    ) -> list[str]:
        """收集当前执行计划中已完成的步骤。"""

        step_results = state.get("step_results", {})
        if not isinstance(step_results, dict):
            return []

        return [step.step_id for step in execution_plan.steps if step.step_id in step_results]

    @staticmethod
    def _select_current_step(
        execution_plan: ExecutionPlan,
        runnable_steps: list[ExecutionStep],
    ) -> ExecutionStep:
        """在可执行步骤中选出当前轮次优先执行的步骤。"""

        runnable_step_ids = {step.step_id for step in runnable_steps}
        for step in execution_plan.steps:
            if step.step_id not in runnable_step_ids:
                continue
            if step.executor != "answer":
                return step
        return runnable_steps[0]
