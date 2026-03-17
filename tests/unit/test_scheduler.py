"""调度节点与调度器单元测试。"""

import pytest

from app.agent.nodes.scheduler_node import SchedulerNode
from app.agent.scheduler import SchedulerService
from app.agent.state import ExecutionPlan, ExecutionStep, ExecutorResult


def test_scheduler_prefers_first_non_answer_runnable_step() -> None:
    """存在多个可执行步骤时，应优先选择首个非 answer 步骤。"""

    scheduler = SchedulerService()
    scheduled_execution = scheduler.schedule(
        {
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="multi_step",
                recommended_route="route",
                steps=[
                    ExecutionStep(
                        step_id="rag_1",
                        executor="rag",
                        goal="检索相关政策",
                        can_run_in_parallel=True,
                    ),
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="查询路线方案",
                        can_run_in_parallel=True,
                    ),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="总结路线与政策结果",
                        depends_on=["rag_1", "route_1"],
                    ),
                ],
            )
        }
    )

    assert scheduled_execution.route == "ragflow"
    assert scheduled_execution.current_step_id == "rag_1"
    assert scheduled_execution.runnable_step_ids == ["rag_1", "route_1"]
    assert scheduled_execution.completed_step_ids == []
    assert scheduled_execution.pending_step_ids == ["rag_1", "route_1", "answer_1"]


def test_scheduler_routes_to_answer_after_dependencies_are_completed() -> None:
    """依赖完成后应把 answer 步骤调度为当前步骤。"""

    scheduler = SchedulerService()
    scheduled_execution = scheduler.schedule(
        {
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="multi_step",
                recommended_route="route",
                steps=[
                    ExecutionStep(step_id="rag_1", executor="rag", goal="检索相关政策"),
                    ExecutionStep(step_id="route_1", executor="route", goal="查询路线方案"),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="总结路线与政策结果",
                        depends_on=["rag_1", "route_1"],
                    ),
                ],
            ),
            "step_results": {
                "rag_1": ExecutorResult(
                    step_id="rag_1",
                    executor="rag",
                    is_success=True,
                    summary="知识检索完成。",
                ),
                "route_1": ExecutorResult(
                    step_id="route_1",
                    executor="route",
                    is_success=True,
                    summary="路线查询完成。",
                ),
            },
        }
    )

    assert scheduled_execution.route == "answer"
    assert scheduled_execution.current_step_id == "answer_1"
    assert scheduled_execution.completed_step_ids == ["rag_1", "route_1"]
    assert scheduled_execution.pending_step_ids == ["answer_1"]


@pytest.mark.asyncio
async def test_scheduler_node_exposes_state_fields() -> None:
    """调度节点应把调度结果写回图状态。"""

    scheduler_node = SchedulerNode()
    result = await scheduler_node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
                steps=[
                    ExecutionStep(
                        step_id="traffic_1",
                        executor="traffic",
                        goal="查询实时路况",
                    ),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="总结路况结果",
                        depends_on=["traffic_1"],
                    ),
                ],
            )
        }
    )

    assert result["scheduled_route"] == "traffic"
    assert result["current_step_id"] == "traffic_1"
    assert result["runnable_step_ids"] == ["traffic_1"]
    assert result["pending_step_ids"] == ["traffic_1", "answer_1"]
