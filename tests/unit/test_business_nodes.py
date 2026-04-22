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
                                "exitInfos": [
                                    {
                                        "tollName": "杭州南收费站",
                                        "entranceStatus": 0,
                                        "exportStatus": 10203,
                                    }
                                ],
                                "trafficCongestions": [
                                    {
                                        "id": "cg-1",
                                        "beginMilestone": 120,
                                        "endMilestone": 128,
                                        "directionType": "1",
                                        "des": "金华方向缓行",
                                        "beginTime": "2026-04-15 08:00:00",
                                        "controlMeasures": "借道通行",
                                        "eventType": "congestion",
                                    }
                                ],
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
    assert result["step_results"]["route_1"].normalized_result["exit_count"] == 1
    assert result["step_results"]["route_1"].normalized_result["congestion_count"] == 1
    assert result["step_results"]["route_1"].normalized_result["exit_items"][0]["toll_name"] == "杭州南收费站"
    assert result["step_results"]["route_1"].normalized_result["congestion_items"][0]["description"] == "金华方向缓行"
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
    assert "查询时间" in result["report_context"]
    assert result["step_results"]["report_1"].executor == "report"
    assert result["step_results"]["report_1"].normalized_result["congestion_total_mile"] == 12.5
    assert result["step_results"]["report_1"].normalized_result["congestion_top_items"][0]["roadName"] == "沪昆高速"
    assert result["step_results"]["report_1"].normalized_result["accident_top_items"][0]["roadName"] == "杭州绕城高速"
    assert result["step_results"]["report_1"].normalized_result["control_top_items"][0]["roadName"] == "长深高速"
class _MultiRouteToolRegistry:
    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name == "live_driving_query":
            return dumps(
                {
                    "routesCount": 2,
                    "routes": [
                        {
                            "distance": 180000,
                            "duration": 120,
                            "toll": 85,
                            "sections": [
                                {"roadName": "杭金衢高速", "trafficControls": [], "serviceAreas": []},
                                {"roadName": "沪昆高速", "trafficControls": [], "serviceAreas": []},
                            ],
                        },
                        {
                            "distance": 195000,
                            "duration": 135,
                            "toll": 78,
                            "sections": [
                                {"roadName": "杭州绕城高速", "trafficControls": [], "serviceAreas": []},
                                {"roadName": "长深高速", "trafficControls": [], "serviceAreas": []},
                            ],
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if tool_name == "live_road_event_query":
            road = str(arguments.get("road") or "")
            payload_by_road = {
                "杭金衢高速": [
                    {
                        "roadName": "杭金衢高速",
                        "roadGbCode": "G60",
                        "congestionInfoList": [
                            {
                                "id": "cg-1",
                                "des": "金华方向缓行",
                                "beginTime": "2026-04-15 08:00:00",
                                "expectedEndTime": "2026-04-15 10:30:00",
                                "beginMilestone": 120,
                                "endMilestone": 128,
                                "directionType": "1",
                                "controlMeasures": "借道通行",
                                "eventType": "congestion",
                                "subEventType": "slow",
                                "roadAmbleMile": 8.0,
                            }
                        ],
                        "trafficControlList": [
                            {
                                "id": "tc-1",
                                "des": "施工占道",
                                "beginTime": "2026-04-15 07:30:00",
                                "expectedEndTime": "2026-04-15 18:00:00",
                                "beginMilestone": 122,
                                "endMilestone": 124,
                                "directionType": "1",
                                "controlMeasures": "封闭第一车道",
                                "eventType": "construction",
                                "subEventType": "lane_closure",
                            }
                        ],
                        "serviceAreaList": [],
                        "exitInfoList": [],
                    }
                ],
                "沪昆高速": [
                    {
                        "roadName": "沪昆高速",
                        "roadGbCode": "G60",
                        "congestionInfoList": [],
                        "trafficControlList": [],
                        "serviceAreaList": [],
                        "exitInfoList": [
                            {
                                "tollName": "诸暨北收费站",
                                "tollId": 101,
                                "entranceStatus": 0,
                                "exportStatus": 10203,
                            }
                        ],
                    }
                ],
                "杭州绕城高速": [
                    {
                        "roadName": "杭州绕城高速",
                        "roadGbCode": "G2504",
                        "congestionInfoList": [
                            {
                                "id": "cg-2",
                                "des": "北向拥堵",
                                "beginTime": "2026-04-15 09:10:00",
                                "expectedEndTime": "2026-04-15 11:00:00",
                                "beginMilestone": 32,
                                "endMilestone": 36,
                                "directionType": "1",
                                "controlMeasures": "间歇放行",
                            }
                        ],
                        "trafficControlList": [],
                        "serviceAreaList": [],
                        "exitInfoList": [],
                    }
                ],
                "长深高速": [
                    {
                        "roadName": "长深高速",
                        "roadGbCode": "G25",
                        "congestionInfoList": [],
                        "trafficControlList": [
                            {
                                "id": "tc-2",
                                "des": "入口限流",
                                "beginTime": "2026-04-15 08:20:00",
                                "expectedEndTime": "2026-04-15 12:00:00",
                                "beginMilestone": 210,
                                "endMilestone": 212,
                                "directionType": "2",
                                "controlMeasures": "货车分批放行",
                                "eventType": "control",
                            }
                        ],
                        "serviceAreaList": [],
                        "exitInfoList": [],
                    }
                ],
            }
            return dumps(payload_by_road.get(road, []), ensure_ascii=False)
        raise AssertionError(f"unexpected tool: {tool_name}")


@pytest.mark.asyncio
async def test_route_node_keeps_all_routes_for_od_queries() -> None:
    node = RouteNode(tool_registry=_MultiRouteToolRegistry())

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={"origin": "杭州", "destination": "金华", "travel_mode": "auto"},
            ),
        }
    )

    normalized_result = result["step_results"]["route_1"].normalized_result
    assert normalized_result["routes_count"] == 2
    assert len(normalized_result["route_summaries"]) == 2
    assert normalized_result["road_names"] == [
        "杭金衢高速",
        "沪昆高速",
        "杭州绕城高速",
        "长深高速",
    ]


@pytest.mark.asyncio
async def test_traffic_node_builds_route_level_event_and_control_details() -> None:
    route_node = RouteNode(tool_registry=_MultiRouteToolRegistry())
    route_result = await route_node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="multi_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={"origin": "杭州", "destination": "金华", "travel_mode": "auto"},
            ),
        }
    )

    traffic_node = TrafficNode(tool_registry=_MultiRouteToolRegistry())
    traffic_result = await traffic_node.run(
        {
            "current_step_id": "traffic_1",
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="multi_step",
                recommended_route="traffic",
                steps=[
                    ExecutionStep(step_id="route_1", executor="route", goal="查询路线"),
                    ExecutionStep(
                        step_id="traffic_1",
                        executor="traffic",
                        goal="查询所有路线路况",
                        depends_on=["route_1"],
                    ),
                ],
            ),
            "step_results": route_result["step_results"],
            "step_arguments": {
                "traffic_1": ResolvedArguments(
                    category="traffic_status",
                    arguments={"query": "杭州到金华堵不堵"},
                )
            },
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={"query": "杭州到金华堵不堵"},
            ),
        }
    )

    normalized_result = traffic_result["step_results"]["traffic_1"].normalized_result
    assert normalized_result["queried_roads"] == [
        "杭金衢高速",
        "沪昆高速",
        "杭州绕城高速",
        "长深高速",
    ]
    assert normalized_result["route_count"] == 2
    assert normalized_result["event_count"] == 5
    assert normalized_result["road_summaries"][0]["event_items"][0]["event_label"] == "拥堵"
    assert normalized_result["road_summaries"][0]["event_items"][1]["event_label"] == "施工"

    first_route = normalized_result["route_summaries"][0]
    second_route = normalized_result["route_summaries"][1]

    assert first_route["road_names"] == ["杭金衢高速", "沪昆高速"]
    assert first_route["congestion_count"] == 1
    assert first_route["traffic_control_count"] == 1
    assert first_route["road_details"][0]["congestion_items"][0]["description"] == "金华方向缓行"
    assert first_route["road_details"][0]["traffic_control_items"][0]["description"] == "施工占道"
    assert first_route["road_details"][0]["congestion_items"][0]["start_time"] == "2026-04-15 08:00:00"
    assert first_route["road_details"][0]["congestion_items"][0]["end_time"] == "2026-04-15 10:30:00"
    assert first_route["road_details"][0]["congestion_items"][0]["control_measures"] == "借道通行"
    assert first_route["road_details"][0]["congestion_items"][0]["direction_label"] == "上行"
    assert first_route["road_details"][0]["congestion_items"][0]["location_description"] == "方向:上行 桩号:120-128"
    assert first_route["road_details"][0]["traffic_control_items"][0]["control_measures"] == "封闭第一车道"
    assert first_route["road_details"][0]["event_items"][0]["event_label"] == "拥堵"
    assert first_route["road_details"][0]["event_items"][1]["event_label"] == "施工"
    assert first_route["road_details"][1]["exit_items"][0]["toll_name"] == "诸暨北收费站"
    assert first_route["road_details"][1]["exit_items"][0]["export_status"] == 10203
    assert first_route["road_details"][1]["exit_items"][0]["export_status_label"] == "限流"
    assert first_route["road_details"][1]["event_items"][0]["event_label"] == "收费站"
    assert first_route["road_details"][1]["event_items"][0]["export_status_label"] == "限流"
    assert first_route["event_items"][1]["event_label"] == "施工"

    assert second_route["road_names"] == ["杭州绕城高速", "长深高速"]
    assert second_route["congestion_count"] == 1
    assert second_route["traffic_control_count"] == 1
    assert second_route["road_details"][0]["congestion_items"][0]["description"] == "北向拥堵"
    assert second_route["road_details"][0]["congestion_items"][0]["direction_label"] == "上行"
    assert second_route["road_details"][1]["traffic_control_items"][0]["description"] == "入口限流"
    assert second_route["road_details"][1]["traffic_control_items"][0]["control_measures"] == "货车分批放行"
    assert second_route["road_details"][1]["event_items"][0]["event_label"] == "管制"
