"""参数提取节点单元测试。"""

import pytest

from app.agent.nodes.argument_node import ArgumentNode
from app.agent.state import ExecutionPlan, ExecutionStep
from app.clients.llm_client import LlmInputMessage


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
                recommended_route="route",
            ),
        }
    )

    resolved_arguments = result["resolved_arguments"]
    assert resolved_arguments.arguments["origin"] == "杭州"
    assert resolved_arguments.arguments["destination"] == "金华"
    assert resolved_arguments.arguments["travel_mode"] == "auto"
    assert result["step_arguments"]["route_1"].arguments["origin"] == "杭州"
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
                recommended_route="route",
            ),
        }
    )

    resolved_arguments = result["resolved_arguments"]
    assert resolved_arguments.missing_fields == ["origin", "destination"]
    assert result["need_clarification"] is True
    assert "起点" in result["clarification_question"]
    assert "终点" in result["clarification_question"]


@pytest.mark.asyncio
async def test_argument_node_extracts_report_query() -> None:
    """路网报告问题应保留原始查询文本供 answer 阶段自行判断。"""

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
    assert resolved_arguments.arguments["query"] == "请基于上次结果生成今天全路网路况对比表格"


@pytest.mark.asyncio
async def test_argument_node_extracts_service_keywords() -> None:
    """服务区问题应提取服务区或设施关键词。"""

    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "杭州东服务区充电桩情况怎么样？",
            "primary_category": "service_area",
            "execution_plan": ExecutionPlan(
                primary_category="service_area",
                execution_mode="single_step",
                recommended_route="service",
            ),
        }
    )

    resolved_arguments = result["resolved_arguments"]
    assert resolved_arguments.arguments["query"] == "杭州东服务区充电桩情况怎么样？"
    assert "杭州东" in resolved_arguments.arguments["keyword"]
    assert result["step_arguments"]["service_1"].category == "service_area"


@pytest.mark.asyncio
async def test_argument_node_builds_step_arguments_for_multi_step_route_plan() -> None:
    """多步骤路线计划应为不同 step 生成独立参数。"""

    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "杭州到金华怎么走，并说明是否符合高速清障标准以及当前路况如何？",
            "primary_category": "route_planning",
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="multi_step",
                recommended_route="route",
                steps=[
                    ExecutionStep(
                        step_id="rag_1",
                        executor="rag",
                        goal="检索路线相关政策和标准要求",
                    ),
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="查询路线规划相关数据",
                    ),
                    ExecutionStep(
                        step_id="traffic_1",
                        executor="traffic",
                        goal="查询路线相关路况信息",
                    ),
                ],
            ),
        }
    )

    step_arguments = result["step_arguments"]
    assert step_arguments["rag_1"].category == "policy"
    assert "高速清障标准" in step_arguments["rag_1"].arguments["query"]
    assert step_arguments["route_1"].arguments["origin"] == "杭州"
    assert step_arguments["route_1"].arguments["destination"] == "金华"
    assert step_arguments["traffic_1"].category == "traffic_status"


@pytest.mark.asyncio
async def test_argument_node_extracts_reference_answer_for_report_requests() -> None:
    """报表类问题带上次回答时，应提取参考回答文本。"""

    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "请基于上次结果生成今天全路网路况对比表格",
            "input_messages": [
                LlmInputMessage(
                    role="assistant",
                    content="上次报告显示杭州北向拥堵指数为 2.1。",
                )
            ],
            "primary_category": "network_report",
            "execution_plan": ExecutionPlan(
                primary_category="network_report",
                execution_mode="single_step",
                recommended_route="report",
                steps=[
                    ExecutionStep(
                        step_id="report_1",
                        executor="report",
                        goal="汇总路网数据",
                    )
                ],
            ),
        }
    )

    assert result["step_arguments"]["report_1"].arguments["reference_answer"] == (
        "上次报告显示杭州北向拥堵指数为 2.1。"
    )


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
