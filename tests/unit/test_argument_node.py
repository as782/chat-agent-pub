"""参数提取节点单元测试。"""

import pytest

from app.agent.nodes.argument_node import ArgumentNode
from app.agent.state import ExecutionPlan


@pytest.mark.asyncio
async def test_argument_node_extracts_route_arguments() -> None:
    """路线问题应提取起点、终点和出行方式。"""

    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "杭州到金华怎么走",
            "primary_category": "route_planning",
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="mcp",
            ),
        }
    )

    resolved_arguments = result["resolved_arguments"]
    assert resolved_arguments.arguments["origin"] == "杭州"
    assert resolved_arguments.arguments["destination"] == "金华"
    assert resolved_arguments.arguments["travel_mode"] == "auto"
    assert result["need_clarification"] is False


@pytest.mark.asyncio
async def test_argument_node_marks_missing_route_arguments() -> None:
    """无法识别起终点时应进入澄清状态。"""

    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "怎么去金华",
            "primary_category": "route_planning",
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="mcp",
            ),
        }
    )

    resolved_arguments = result["resolved_arguments"]
    assert resolved_arguments.missing_fields == ["origin", "destination"]
    assert result["need_clarification"] is True
    assert "起点" in result["clarification_question"]
    assert "终点" in result["clarification_question"]


@pytest.mark.asyncio
async def test_argument_node_extracts_report_flags() -> None:
    """路网报告问题应提取表格和对比需求。"""

    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "请基于上次结果生成今天全路网路况对比表格",
            "primary_category": "network_report",
            "execution_plan": ExecutionPlan(
                primary_category="network_report",
                execution_mode="single_step",
                recommended_route="answer",
            ),
        }
    )

    resolved_arguments = result["resolved_arguments"]
    assert resolved_arguments.arguments["scope"] == "全路网"
    assert resolved_arguments.arguments["need_table"] is True
    assert resolved_arguments.arguments["need_comparison"] is True


@pytest.mark.asyncio
async def test_argument_node_extracts_policy_query() -> None:
    """政策类问题应去掉知识库前缀并保留检索 query。"""

    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "knowledge: 高速清障最低标准是什么？",
            "primary_category": "policy",
            "execution_plan": ExecutionPlan(
                primary_category="policy",
                execution_mode="single_step",
                recommended_route="ragflow",
            ),
        }
    )

    resolved_arguments = result["resolved_arguments"]
    assert resolved_arguments.arguments["query"] == "高速清障最低标准是什么？"
