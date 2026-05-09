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
                    "congestionTopN": [
                        {
                            "id": "cg-1",
                            "roadGBCode": "G60",
                            "roadName": "沪昆高速",
                            "directionType": "1",
                            "beginMilestone": 120,
                            "endMilestone": 128,
                            "roadAmbleMile": 8.0,
                        }
                    ],
                    "controlTopN": [
                        {
                            "id": "ct-1",
                            "roadGBCode": "G25",
                            "roadName": "长深高速",
                            "direction": "0",
                            "tollName": "诸暨北收费站",
                            "startTime": "2026-03-31 08:10:00",
                        }
                    ],
                    "exitTopN": [
                        {
                            "tollId": 1029,
                            "tollName": "萧山收费站",
                            "entranceStatus": 0,
                            "exportStatus": 10202,
                        }
                    ],
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
async def test_service_node_prefers_named_service_area_over_direction_query_terms() -> None:
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
                    "query": "\u957f\u5b89\u670d\u52a1\u533a",
                    "keyword": "\u957f\u5b89\u670d\u52a1\u533a",
                    "service_name": "\u957f\u5b89\u670d\u52a1\u533a",
                    "catalog_service_match": True,
                    "service_query_terms": [
                        "\u957f\u5b89\u670d\u52a1\u533a\u5317\u533a",
                        "\u957f\u5b89\u670d\u52a1\u533a\u676d\u5dde\u65b9\u5411",
                        "\u957f\u5b89\u670d\u52a1\u533a",
                    ],
                },
            ),
        }
    )

    assert tool_registry.calls == [
        {
            "tool_name": "live_service_query",
            "arguments": {"keyword": "\u957f\u5b89\u670d\u52a1\u533a"},
        }
    ]


@pytest.mark.asyncio
async def test_service_node_skips_tool_for_unknown_named_service_area() -> None:
    class _CapturingServiceToolRegistry(_FakeToolRegistry):
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
            self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
            return await super().execute_named_tool(tool_name=tool_name, arguments=arguments)

    tool_registry = _CapturingServiceToolRegistry()
    node = ServiceNode(tool_registry=tool_registry)

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="service_area",
                execution_mode="single_step",
                recommended_route="service",
            ),
            "resolved_arguments": ResolvedArguments(
                category="service_area",
                arguments={
                    "query": "龙游服务区状态",
                    "keyword": "龙游服务区",
                    "service_name": "龙游服务区",
                    "catalog_service_match": False,
                },
            ),
        }
    )

    assert tool_registry.calls == []
    assert "不在集团管辖范围内" in result["service_context"]
    normalized_result = result["step_results"]["service_1"].normalized_result
    assert normalized_result["result_count"] == 0
    assert normalized_result["group_scope_miss"] is True
    assert normalized_result["catalog_service_match"] is False


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
    report_result = result["step_results"]["report_1"].normalized_result
    assert result["step_results"]["report_1"].executor == "report"
    assert report_result["congestion_total_mile"] == 12.5
    assert report_result["congestion_top_items"][0]["roadName"] == "沪昆高速"
    assert report_result["control_top_items"][0]["roadName"] == "长深高速"
    assert report_result["exit_top_count"] == 1
    assert report_result["exit_top_items"][0]["tollName"] == "萧山收费站"
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


class _DirectionalFilterToolRegistry:
    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name == "live_driving_query":
            return dumps(
                {
                    "routesCount": 1,
                    "routes": [
                        {
                            "distance": 120000,
                            "duration": 95,
                            "toll": 42,
                            "sections": [
                                {
                                    "roadName": "沪昆高速",
                                    "trafficCongestions": [
                                        {
                                            "id": "cg-hz",
                                            "directionType": "02",
                                            "des": "G60沪昆高速-杭州方向，过直埠枢纽5.4公里缓行",
                                            "beginMilestone": 212711,
                                            "endMilestone": 212900,
                                            "beginTime": "2026-04-23 10:07:48",
                                        },
                                        {
                                            "id": "cg-jx",
                                            "directionType": "01",
                                            "des": "G60沪昆高速-江西方向，过后宅枢纽1.9公里缓行",
                                            "beginMilestone": 272600,
                                            "endMilestone": 272900,
                                            "beginTime": "2026-04-23 10:08:00",
                                        },
                                        {
                                            "id": "cg-bi",
                                            "directionType": "00",
                                            "des": "G60沪昆高速-双向，张家畈枢纽附近养护占道",
                                            "beginMilestone": 185728,
                                            "endMilestone": 203780,
                                            "beginTime": "2026-04-23 10:03:12",
                                        },
                                    ],
                                    "trafficControls": [
                                        {
                                            "id": "tc-hz",
                                            "directionType": "02",
                                            "des": (
                                                "G60沪昆高速-杭州方向，"
                                                "过直埠枢纽5.4公里抛锚占道"
                                            ),
                                            "beginMilestone": 212711,
                                            "endMilestone": 212711,
                                            "beginTime": "2026-04-23 10:07:48",
                                        },
                                        {
                                            "id": "tc-jx",
                                            "directionType": "01",
                                            "des": (
                                                "G60沪昆高速-江西方向，"
                                                "过后宅枢纽0.9公里事故占道"
                                            ),
                                            "beginMilestone": 271600,
                                            "endMilestone": 271600,
                                            "beginTime": "2026-04-23 10:20:38",
                                        },
                                        {
                                            "id": "tc-bi",
                                            "directionType": "00",
                                            "des": "G60沪昆高速-双向，次坞收费站到直埠枢纽之间施工",
                                            "beginMilestone": 209500,
                                            "endMilestone": 256800,
                                            "beginTime": "2026-04-23 09:04:24",
                                        },
                                    ],
                                    "serviceAreas": [{"serviceName": "诸暨服务区"}],
                                    "exitInfos": [
                                        {
                                            "tollName": "次坞收费站",
                                            "entranceStatus": 0,
                                            "exportStatus": 0,
                                        }
                                    ],
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
                        "roadName": "沪昆高速",
                        "roadGbCode": "G60",
                        "congestionInfoList": [
                            {
                                "id": "cg-hz",
                                "directionType": "02",
                                "des": (
                                    "G60沪昆高速（杭金衢）-杭州方向，"
                                    "过直埠枢纽5.4公里发生车辆故障（抛锚）"
                                ),
                                "beginTime": "2026-04-23 10:07:48",
                                "beginMilestone": 212711,
                                "endMilestone": 212711,
                            },
                            {
                                "id": "cg-jx",
                                "directionType": "01",
                                "des": "G60沪昆高速（杭金衢）-江西方向，过后宅枢纽1.9公里发生缓行",
                                "beginTime": "2026-04-23 10:08:00",
                                "beginMilestone": 272600,
                                "endMilestone": 272900,
                            },
                            {
                                "id": "cg-bi",
                                "directionType": "00",
                                "des": "G60沪昆高速（杭金衢）-双向，张家畈枢纽附近养护占道",
                                "beginTime": "2026-04-23 10:03:12",
                                "beginMilestone": 185728,
                                "endMilestone": 203780,
                            },
                        ],
                        "trafficControlList": [
                            {
                                "id": "tc-hz",
                                "directionType": "02",
                                "des": (
                                    "G60沪昆高速（杭金衢）-杭州方向，"
                                    "过直埠枢纽5.4公里发生车辆故障（抛锚）"
                                ),
                                "beginTime": "2026-04-23 10:07:48",
                                "beginMilestone": 212711,
                                "endMilestone": 212711,
                                "eventType": "control",
                            },
                            {
                                "id": "tc-jx",
                                "directionType": "01",
                                "des": (
                                    "G60沪昆高速（杭金衢）-江西方向，"
                                    "过后宅枢纽0.9公里发生交通事故（追尾）"
                                ),
                                "beginTime": "2026-04-23 10:20:38",
                                "beginMilestone": 271600,
                                "endMilestone": 271600,
                                "eventType": "control",
                            },
                            {
                                "id": "tc-bi",
                                "directionType": "00",
                                "des": (
                                    "G60沪昆高速（杭金衢）-双向，"
                                    "在次坞收费站和直埠枢纽之间发生道路施工"
                                ),
                                "beginTime": "2026-04-23 09:04:24",
                                "beginMilestone": 209500,
                                "endMilestone": 256800,
                                "eventType": "construction",
                            },
                        ],
                        "serviceAreaList": [],
                        "exitInfoList": [],
                    }
                ],
                ensure_ascii=False,
            )
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


@pytest.mark.asyncio
async def test_route_node_filters_opposite_direction_events_for_od_queries() -> None:
    node = RouteNode(tool_registry=_DirectionalFilterToolRegistry())

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={"origin": "杭州", "destination": "义乌", "travel_mode": "auto"},
            ),
        }
    )

    normalized_result = result["step_results"]["route_1"].normalized_result
    congestion_descriptions = [
        item["description"] for item in normalized_result["congestion_items"]
    ]
    control_descriptions = [
        item["description"] for item in normalized_result["traffic_controls"]
    ]

    assert "G60沪昆高速-杭州方向，过直埠枢纽5.4公里缓行" not in congestion_descriptions
    assert "G60沪昆高速-江西方向，过后宅枢纽1.9公里缓行" in congestion_descriptions
    assert "G60沪昆高速-双向，张家畈枢纽附近养护占道" in congestion_descriptions

    assert "G60沪昆高速-杭州方向，过直埠枢纽5.4公里抛锚占道" not in control_descriptions
    assert "G60沪昆高速-江西方向，过后宅枢纽0.9公里事故占道" in control_descriptions
    assert "G60沪昆高速-双向，次坞收费站到直埠枢纽之间施工" in control_descriptions
    assert "直埠枢纽5.4公里抛锚占道" not in result["route_context"]


@pytest.mark.asyncio
async def test_traffic_node_filters_opposite_direction_events_for_od_queries() -> None:
    node = TrafficNode(tool_registry=_DirectionalFilterToolRegistry())

    result = await node.run(
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
                        goal="查询路线相关路况",
                        depends_on=["route_1"],
                    ),
                ],
            ),
            "step_results": {
                "route_1": ExecutorResult(
                    step_id="route_1",
                    executor="route",
                    is_success=True,
                    normalized_result={
                        "origin": "杭州",
                        "destination": "义乌",
                        "road_names": ["沪昆高速"],
                        "route_summaries": [{"route_index": 1, "road_names": ["沪昆高速"]}],
                    },
                )
            },
            "step_arguments": {
                "traffic_1": ResolvedArguments(
                    category="traffic_status",
                    arguments={"query": "杭州到义乌堵不堵"},
                )
            },
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={"query": "杭州到义乌堵不堵"},
            ),
        }
    )

    normalized_result = result["step_results"]["traffic_1"].normalized_result
    congestion_descriptions = [
        item["description"] for item in normalized_result["congestion_items"]
    ]
    control_descriptions = [
        item["description"] for item in normalized_result["traffic_control_items"]
    ]

    assert (
        "G60沪昆高速（杭金衢）-杭州方向，过直埠枢纽5.4公里发生车辆故障（抛锚）"
        not in congestion_descriptions
    )
    assert "G60沪昆高速（杭金衢）-江西方向，过后宅枢纽1.9公里发生缓行" in congestion_descriptions
    assert "G60沪昆高速（杭金衢）-双向，张家畈枢纽附近养护占道" in congestion_descriptions

    assert (
        "G60沪昆高速（杭金衢）-杭州方向，过直埠枢纽5.4公里发生车辆故障（抛锚）"
        not in control_descriptions
    )
    assert (
        "G60沪昆高速（杭金衢）-江西方向，过后宅枢纽0.9公里发生交通事故（追尾）"
        in control_descriptions
    )
    assert "G60沪昆高速（杭金衢）-双向，在次坞收费站和直埠枢纽之间发生道路施工" in control_descriptions


def test_route_extractors_filter_vehicle_fault_events() -> None:
    sections = [
        {
            "roadName": "S26诸永高速",
            "trafficControls": [
                {"id": "vehicle-fault-control", "eventType": "97", "des": "车辆故障"},
                {"id": "construction", "eventType": "05", "des": "道路施工"},
            ],
            "trafficCongestions": [
                {"id": "vehicle-fault-congestion", "eventType": "97", "des": "车辆故障"},
                {"id": "slow", "eventType": "105", "des": "道路缓行"},
            ],
        }
    ]

    traffic_controls = RouteNode._extract_traffic_controls(sections)
    route_control_items = RouteNode._extract_route_control_items(sections)
    congestion_items = RouteNode._extract_congestion_items(sections)

    assert [item["control_id"] for item in traffic_controls] == ["construction"]
    assert [item["description"] for item in route_control_items] == ["道路施工"]
    assert [item["congestion_id"] for item in congestion_items] == ["slow"]
