"""业务节点单元测试。"""

import pytest

from app.agent.nodes.report_node import ReportNode
from app.agent.nodes.route_node import RouteNode
from app.agent.nodes.traffic_node import TrafficNode
from app.agent.state import ExecutionPlan, ResolvedArguments


@pytest.mark.asyncio
async def test_route_node_builds_business_context() -> None:
    """路线节点应把结构化参数整理为上下文文本。"""

    node = RouteNode()

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={
                    "origin": "杭州",
                    "destination": "金华",
                    "travel_mode": "auto",
                },
            ),
        }
    )

    assert result["route_context"] is not None
    assert "杭州" in result["route_context"]
    assert result["step_results"]["route_1"].executor == "route"
    assert result["step_results"]["route_1"].normalized_result["destination"] == "金华"


@pytest.mark.asyncio
async def test_route_node_prefers_step_specific_arguments() -> None:
    """存在逐步参数时，路线节点应优先使用当前 step 的参数。"""

    node = RouteNode()

    result = await node.run(
        {
            "current_step_id": "route_1",
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="multi_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={"origin": "错误起点", "destination": "错误终点"},
            ),
            "step_arguments": {
                "route_1": ResolvedArguments(
                    category="route_planning",
                    arguments={
                        "origin": "杭州",
                        "destination": "金华",
                        "travel_mode": "auto",
                    },
                )
            },
        }
    )

    assert "杭州" in result["route_context"]
    assert result["step_results"]["route_1"].normalized_result["destination"] == "金华"


@pytest.mark.asyncio
async def test_traffic_node_builds_business_context() -> None:
    """路况节点应把结构化参数整理为上下文文本。"""

    node = TrafficNode()

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={"target": "杭金衢高速", "time_range": "current"},
            ),
        }
    )

    assert result["traffic_context"] is not None
    assert "杭金衢高速" in result["traffic_context"]
    assert result["step_results"]["traffic_1"].executor == "traffic"
    assert result["step_results"]["traffic_1"].normalized_result["target"] == "杭金衢高速"


@pytest.mark.asyncio
async def test_report_node_builds_business_context() -> None:
    """报表节点应把结构化参数整理为上下文文本。"""

    node = ReportNode()

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="network_report",
                execution_mode="single_step",
                recommended_route="report",
            ),
            "resolved_arguments": ResolvedArguments(
                category="network_report",
                arguments={"scope": "全路网", "need_table": True},
            ),
        }
    )

    assert result["report_context"] is not None
    assert "全路网" in result["report_context"]
    assert result["step_results"]["report_1"].executor == "report"
    assert result["step_results"]["report_1"].normalized_result["scope"] == "全路网"
