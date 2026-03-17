"""Planner 服务单元测试。"""

from app.agent.planner import PlannerService
from app.agent.state import AgentState


def test_planner_returns_tool_plan_when_tools_are_explicitly_requested() -> None:
    """显式传入 tools 时，应优先生成工具执行计划。"""

    planner = PlannerService()

    plan = planner.build_plan(
        AgentState(
            latest_user_message="请帮我计算 1+1",
            requested_tool_names=["calculator"],
        )
    )

    assert plan.primary_category == "general"
    assert plan.recommended_route == "tool"
    assert [step.executor for step in plan.steps] == ["tool", "answer"]


def test_planner_marks_policy_requests() -> None:
    """政策类问题应生成知识检索计划。"""

    planner = PlannerService()

    plan = planner.build_plan(AgentState(latest_user_message="knowledge: 高速清障最低标准是什么？"))

    assert plan.primary_category == "policy"
    assert plan.recommended_route == "ragflow"
    assert [step.executor for step in plan.steps] == ["rag", "answer"]


def test_planner_marks_route_planning_requests() -> None:
    """路线规划类问题应生成 MCP 路线查询计划。"""

    planner = PlannerService()

    plan = planner.build_plan(AgentState(latest_user_message="杭州到金华怎么走"))

    assert plan.primary_category == "route_planning"
    assert plan.recommended_route == "route"
    assert [step.executor for step in plan.steps] == ["route", "answer"]


def test_planner_builds_multi_step_route_and_policy_plan() -> None:
    """路线类问题包含政策约束时，应拆成知识检索与路线查询的多步骤计划。"""

    planner = PlannerService()

    plan = planner.build_plan(
        AgentState(latest_user_message="杭州到金华怎么走，并说明是否符合高速清障标准？")
    )

    assert plan.primary_category == "route_planning"
    assert plan.execution_mode == "multi_step"
    assert plan.recommended_route == "route"
    assert [step.executor for step in plan.steps] == ["rag", "route", "answer"]


def test_planner_builds_multi_step_route_and_traffic_plan() -> None:
    """路线类问题带路况诉求时，应拆成路线与路况两个执行步骤。"""

    planner = PlannerService()

    plan = planner.build_plan(AgentState(latest_user_message="杭州到金华怎么走，当前路况怎么样？"))

    assert plan.primary_category == "route_planning"
    assert plan.execution_mode == "multi_step"
    assert plan.recommended_route == "route"
    assert [step.executor for step in plan.steps] == ["route", "traffic", "answer"]


def test_planner_marks_traffic_status_requests() -> None:
    """路况类问题应生成交通数据查询计划。"""

    planner = PlannerService()

    plan = planner.build_plan(AgentState(latest_user_message="当前杭金衢高速路况怎么样？"))

    assert plan.primary_category == "traffic_status"
    assert plan.recommended_route == "traffic"
    assert [step.executor for step in plan.steps] == ["traffic", "answer"]


def test_planner_builds_multi_step_traffic_and_policy_plan() -> None:
    """路况问题带政策约束时，应拆成知识检索与路况查询两个步骤。"""

    planner = PlannerService()

    plan = planner.build_plan(
        AgentState(latest_user_message="当前杭金衢高速路况怎么样，是否符合清障标准？")
    )

    assert plan.primary_category == "traffic_status"
    assert plan.execution_mode == "multi_step"
    assert plan.recommended_route == "traffic"
    assert [step.executor for step in plan.steps] == ["rag", "traffic", "answer"]


def test_planner_marks_network_report_requests() -> None:
    """全路网汇总和报表类问题应生成 report 计划。"""

    planner = PlannerService()

    plan = planner.build_plan(
        AgentState(latest_user_message="请基于上次结果做一个今天全路网路况对比表格")
    )

    assert plan.primary_category == "network_report"
    assert plan.execution_mode == "single_step"
    assert plan.recommended_route == "report"
    assert [step.executor for step in plan.steps] == ["report", "answer"]


def test_planner_builds_multi_step_report_and_policy_plan() -> None:
    """路网报告问题带政策要求时，应拆成知识检索与报告汇总两个步骤。"""

    planner = PlannerService()

    plan = planner.build_plan(
        AgentState(
            latest_user_message="请基于上次结果做一个今天全路网路况对比表格，并说明是否符合相关标准"
        )
    )

    assert plan.primary_category == "network_report"
    assert plan.execution_mode == "multi_step"
    assert plan.recommended_route == "report"
    assert [step.executor for step in plan.steps] == ["rag", "report", "answer"]


def test_planner_defaults_to_general_for_plain_questions() -> None:
    """普通问答应回落到 general 计划。"""

    planner = PlannerService()

    plan = planner.build_plan(AgentState(latest_user_message="你好"))

    assert plan.primary_category == "general"
    assert plan.recommended_route == "answer"
    assert [step.executor for step in plan.steps] == ["answer"]
