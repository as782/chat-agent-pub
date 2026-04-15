"""业务节点单元测试。"""

from json import dumps

import pytest

from app.agent.nodes.report_node import ReportNode
from app.agent.nodes.route_node import RouteNode
from app.agent.nodes.service_node import ServiceNode
from app.agent.nodes.traffic_node import TrafficNode
from app.agent.state import ExecutionPlan, ExecutionStep, ExecutorResult, ResolvedArguments


class _FakeToolRegistry:
    """最小工具注册表测试桩。"""

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name == "live_driving_query":
            return dumps(
                {
                    "routesCount": 1,
                    "routes": [
                        {
                            "distance": 180000,
                            "duration": 120,
                            "toll": 85,
                            "sections": [
                                {
                                    "roadName": "杭金衢高速",
                                    "trafficControls": [{"id": "tc-1"}],
                                    "serviceAreas": [{"serviceName": "诸暨服务区"}],
                                },
                                {
                                    "roadName": "沪昆高速",
                                    "trafficControls": [{"id": "tc-2"}],
                                    "serviceAreas": [{"serviceName": "金华服务区"}],
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if tool_name == "live_road_event_query":
            return dumps(
                [
                    {
                        "roadName": "杭金衢高速",
                        "congestionInfoList": [{"id": "cg-1"}],
                        "trafficControlList": [{"id": "tc-1"}],
                        "serviceAreaList": [{"serviceName": "杭州服务区"}],
                        "exitInfoList": [{"tollName": "杭州南"}],
                    }
                ],
                ensure_ascii=False,
            )
        if tool_name == "live_service_query":
            return dumps(
                [
                    {
                        "serviceName": "杭州东服务区",
                        "roadName": "沪昆高速",
                        "statusTag": "繁忙",
                        "chargeList": [{"manufacturerName": "国网"}],
                        "commercialList": [{"name": "便利店"}],
                        "tags": ["餐饮", "休息区"],
                    }
                ],
                ensure_ascii=False,
            )
        if tool_name == "live_network_overview_query":
            return dumps(
                {
                    "queryTime": "2026-03-31 09:00:00",
                    "congestion": {"totalMile": 12.5},
                    "congestionTopN": [{"id": "cg-1", "roadName": "沪昆高速"}],
                    "accidentTopN": [{"id": "ac-1", "roadName": "杭州绕城高速"}],
                    "controlTopN": [{"id": "ct-1", "roadName": "长深高速"}],
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected tool: {tool_name}")


@pytest.mark.asyncio
async def test_route_node_builds_business_context() -> None:
    """路线节点应把结构化参数整理为上下文文本。"""

    node = RouteNode(tool_registry=_FakeToolRegistry())

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
    assert result["step_results"]["route_1"].normalized_result["road_names"] == [
        "杭金衢高速",
        "沪昆高速",
    ]
    assert result["step_results"]["route_1"].normalized_result["service_area_names"] == [
        "诸暨服务区",
        "金华服务区",
    ]
    assert result["step_results"]["route_1"].normalized_result["traffic_controls"][0]["control_id"] == "tc-1"


@pytest.mark.asyncio
async def test_route_node_prefers_step_specific_arguments() -> None:
    """存在逐步参数时，路线节点应优先使用当前 step 的参数。"""

    node = RouteNode(tool_registry=_FakeToolRegistry())

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

    node = TrafficNode(tool_registry=_FakeToolRegistry())

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
    assert result["step_results"]["traffic_1"].normalized_result["road"] == "杭金衢高速"
    assert result["step_results"]["traffic_1"].normalized_result["has_congestion"] is True
    assert result["step_results"]["traffic_1"].normalized_result["matched_road_names"] == ["杭金衢高速"]
    assert result["step_results"]["traffic_1"].normalized_result["exit_items"][0]["toll_name"] == "杭州南"


@pytest.mark.asyncio
async def test_traffic_node_prefers_route_derived_road_names() -> None:
    """当 traffic 步骤依赖 route 时，应优先消费 route 产出的 road_names。"""

    node = TrafficNode(tool_registry=_FakeToolRegistry())

    result = await node.run(
        {
            "current_step_id": "traffic_1",
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="multi_step",
                recommended_route="route",
                steps=[
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="查询路线",
                    ),
                    ExecutionStep(
                        step_id="traffic_1",
                        executor="traffic",
                        goal="查询路况",
                        depends_on=["route_1"],
                    ),
                ],
            ),
            "step_results": {
                "route_1": ExecutorResult(
                    step_id="route_1",
                    executor="route",
                    is_success=True,
                    normalized_result={"road_names": ["杭金衢高速", "沪昆高速"]},
                )
            },
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={"target": "错误道路"},
            ),
            "step_arguments": {
                "traffic_1": ResolvedArguments(
                    category="traffic_status",
                    arguments={"target": "错误道路"},
                )
            },
        }
    )

    assert result["step_results"]["traffic_1"].normalized_result["road"] == "杭金衢高速"
    assert result["step_results"]["traffic_1"].normalized_result["queried_roads"] == [
        "杭金衢高速",
        "沪昆高速",
    ]


@pytest.mark.asyncio
async def test_service_node_builds_business_context() -> None:
    """服务区节点应把接口结果整理为上下文文本。"""

    node = ServiceNode(tool_registry=_FakeToolRegistry())

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="service_area",
                execution_mode="single_step",
                recommended_route="service",
            ),
            "resolved_arguments": ResolvedArguments(
                category="service_area",
                arguments={"keyword": "杭州东服务区"},
            ),
        }
    )

    assert result["service_context"] is not None
    assert "杭州东服务区" in result["service_context"]
    assert result["step_results"]["service_1"].executor == "service"
    assert result["step_results"]["service_1"].normalized_result["service_name"] == "杭州东服务区"
    assert result["step_results"]["service_1"].normalized_result["has_charging"] is True
    assert result["step_results"]["service_1"].normalized_result["charge_items"][0]["brand"] == "国网"
    assert result["step_results"]["service_1"].normalized_result["commercial_items"][0]["name"] == "便利店"
    assert result["step_results"]["service_1"].normalized_result["tags"] == ["餐饮", "休息区"]


@pytest.mark.asyncio
async def test_service_node_prefers_structured_service_name_over_raw_query() -> None:
    class _CapturingServiceToolRegistry(_FakeToolRegistry):
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
            self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
            return await super().execute_named_tool(tool_name=tool_name, arguments=arguments)

    tool_registry = _CapturingServiceToolRegistry()
    node = ServiceNode(tool_registry=tool_registry)

    await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="service_area",
                execution_mode="single_step",
                recommended_route="service",
            ),
            "resolved_arguments": ResolvedArguments(
                category="service_area",
                arguments={
                    "query": "杭州东服务区现在有充电桩吗",
                    "keyword": "杭州东服务区现在有充电桩吗",
                    "service_name": "杭州东服务区",
                    "facility_type": "charging",
                },
            ),
        }
    )

    assert tool_registry.calls == [
        {
            "tool_name": "live_service_query",
            "arguments": {"keyword": "杭州东服务区"},
        }
    ]


@pytest.mark.asyncio
async def test_report_node_builds_business_context() -> None:
    """报表节点应把结构化参数整理为上下文文本。"""

    node = ReportNode(tool_registry=_FakeToolRegistry())

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="network_report",
                execution_mode="single_step",
                recommended_route="report",
            ),
            "resolved_arguments": ResolvedArguments(
                category="network_report",
                arguments={"query": "请生成今天全路网路况对比表格"},
            ),
        }
    )

    assert result["report_context"] is not None
    assert "queryTime" in result["report_context"]
    assert result["step_results"]["report_1"].executor == "report"
    assert result["step_results"]["report_1"].normalized_result["congestion_total_mile"] == 12.5
    assert result["step_results"]["report_1"].normalized_result["congestion_top_items"][0]["roadName"] == "沪昆高速"
    assert result["step_results"]["report_1"].normalized_result["accident_top_items"][0]["roadName"] == "杭州绕城高速"
    assert result["step_results"]["report_1"].normalized_result["control_top_items"][0]["roadName"] == "长深高速"
