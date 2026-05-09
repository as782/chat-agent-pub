from __future__ import annotations

from json import dumps

import pytest

from app.agent.argument_resolver import ArgumentResolver
from app.agent.nodes.traffic_node import TrafficNode
from app.agent.planner import PlannerService
from app.agent.prompts import PLANNER_JSON_OUTPUT_PROMPT, PLANNER_PROMPT
from app.agent.state import ExecutionPlan, ResolvedArguments


def test_argument_resolver_only_extracts_surface_context_for_alias_query() -> None:
    resolver = ArgumentResolver()

    result = resolver.resolve(
        {
            "latest_user_message": "沪杭高速沪向车道全部畅通吗？",
            "primary_category": "traffic_status",
        }
    )

    assert result.arguments["road"] == "沪杭高速"
    assert result.arguments["target"] == "沪杭高速沪向车道全部畅通吗？"
    assert result.arguments["direction"] == "沪向"
    assert "road_code" not in result.arguments
    assert "road_name" not in result.arguments


def test_argument_resolver_only_extracts_station_and_direction_without_mapping() -> None:
    resolver = ArgumentResolver()

    result = resolver.resolve(
        {
            "latest_user_message": "诸暨北收费站温州方向出口堵吗？",
            "primary_category": "traffic_status",
        }
    )

    assert result.arguments["toll_station"] == "诸暨北收费站"
    assert result.arguments["direction"] == "温州方向"
    assert result.arguments["road"] == "诸暨北收费站温州方向出口堵吗？"
    assert "road_code" not in result.arguments
    assert "road_name" not in result.arguments


def test_planner_preserves_llm_inferred_canonical_traffic_metadata() -> None:
    planner = PlannerService()

    metadata = planner._enrich_step_metadata(
        executor="traffic",
        metadata={
            "query": "沪杭高速沪向车道全部畅通吗？",
            "road": "沪昆高速",
            "road_name": "沪昆高速",
            "road_code": "G60",
            "target": "沪向车道",
            "direction": "杭州方向",
            "query_intent": "traffic_status",
        },
        latest_user_message="沪杭高速沪向车道全部畅通吗？",
        primary_category="traffic_status",
    )

    assert metadata["road"] == "G60"
    assert metadata["road_name"] == "沪昆高速"
    assert metadata["road_code"] == "G60"
    assert metadata["direction"] == "杭州方向"
    assert metadata["target"] == "沪向车道"


def test_planner_uses_toll_catalog_when_no_road_hint_is_available() -> None:
    planner = PlannerService()

    metadata = planner._enrich_step_metadata(
        executor="traffic",
        metadata={
            "query": "宁波东收费站堵车吗？",
            "road": "G92杭州湾跨海大桥连接线",
            "target": "宁波东收费站",
            "toll_station": "宁波东收费站",
            "query_intent": "traffic_status",
        },
        latest_user_message="宁波东收费站堵车吗？",
        primary_category="traffic_status",
    )

    assert metadata["road"] == "S1"
    assert metadata["road_name"] == "甬台温高速"
    assert metadata["road_code"] == "S1"
    assert metadata["toll_station"] == "宁波东收费站"


def test_planner_treats_enter_highway_as_status_action_not_road_hint() -> None:
    planner = PlannerService()
    message = "\u676d\u5dde\u6536\u8d39\u7ad9\u73b0\u5728\u80fd\u4e0a\u9ad8\u901f\u5417"
    toll_station = "\u676d\u5dde\u6536\u8d39\u7ad9"

    metadata = planner._enrich_step_metadata(
        executor="traffic",
        metadata={
            "query": message,
            "road": "G60",
            "road_name": "\u6caa\u6606\u9ad8\u901f",
            "road_code": "G60",
            "toll_station": toll_station,
            "target": f"{toll_station}\u5165\u53e3",
            "direction": "\u4e0a\u9ad8\u901f",
            "query_intent": "traffic_status",
        },
        latest_user_message=message,
        primary_category="traffic_status",
    )

    assert metadata["road"] == "S2"
    assert metadata["road_name"] == "\u676d\u752c\u9ad8\u901f"
    assert metadata["road_code"] == "S2"
    assert metadata["toll_station"] == toll_station


def test_planner_keeps_direct_road_identifier_without_toll_override() -> None:
    planner = PlannerService()

    metadata = planner._enrich_step_metadata(
        executor="traffic",
        metadata={
            "query": "G92杭州湾跨海大桥连接线堵吗？",
            "road": "G92杭州湾跨海大桥连接线",
            "road_name": "杭州湾跨海大桥连接线",
            "road_code": "G92",
            "query_intent": "traffic_status",
        },
        latest_user_message="G92杭州湾跨海大桥连接线堵吗？",
        primary_category="traffic_status",
    )

    assert metadata["road"] == "G92"
    assert metadata["road_name"] == "杭州湾跨海大桥连接线"
    assert metadata["road_code"] == "G92"


def test_planner_does_not_override_road_for_generic_toll_station_queries() -> None:
    planner = PlannerService()

    metadata = planner._enrich_step_metadata(
        executor="traffic",
        metadata={
            "query": "G25那个收费站关闭",
            "road": "G25",
            "road_name": "长深高速",
            "road_code": "G25",
            "toll_station": "收费站",
            "target": "收费站关闭",
            "query_intent": "traffic_status",
        },
        latest_user_message="G25那个收费站关闭",
        primary_category="traffic_status",
    )

    assert metadata["road"] == "G25"
    assert metadata["road_name"] == "长深高速"
    assert metadata["road_code"] == "G25"


def test_planner_coerces_multi_road_string_into_roads_list() -> None:
    planner = PlannerService()

    metadata = planner._enrich_step_metadata(
        executor="traffic",
        metadata={
            "query": "G60和S26哪条更堵？",
            "roads": "G60, S26",
            "query_intent": "traffic_status",
        },
        latest_user_message="G60和S26哪条更堵？",
        primary_category="traffic_status",
    )

    assert metadata["roads"] == ["G60", "S26"]
    assert "road" not in metadata
    assert "road_name" not in metadata
    assert "road_code" not in metadata


def test_planner_prompt_requires_llm_to_infer_canonical_road() -> None:
    combined_prompt = f"{PLANNER_PROMPT}\n{PLANNER_JSON_OUTPUT_PROMPT}"

    assert "只要问题中能明确识别出起点和终点，例如“衢州到杭州走杭金衢高速堵不堵”“温州回杭州堵吗”" in combined_prompt
    assert "road: string，只能表示单条道路，且值必须是“纯编号”或“纯名称”二选一" in combined_prompt
    assert "宁波东收费站堵车吗" in combined_prompt
    assert "road_name=标准高速名称、road_code=标准高速编号" in combined_prompt
    assert "road 必须优先填写纯编号" in combined_prompt


class _CapturingToolRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        return dumps(
            [
                {
                    "roadName": "沪昆高速",
                    "congestionInfoList": [],
                    "trafficControlList": [],
                    "serviceAreaList": [],
                    "exitInfoList": [{"tollName": "诸暨北收费站"}],
                }
            ],
            ensure_ascii=False,
        )


class _MismatchedRoadToolRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        road = arguments.get("road")
        if road == "S32":
            road_name = "Qianhuang Expressway"
            road_code = "S32"
        else:
            road_name = "Shenjiahu Expressway"
            road_code = "S12"
        return dumps(
            [
                {
                    "roadName": road_name,
                    "roadGbCode": road_code,
                    "congestionInfoList": [],
                    "trafficControlList": [],
                    "serviceAreaList": [],
                    "exitInfoList": [],
                }
            ],
            ensure_ascii=False,
        )


class _EmptyNameRetryToolRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        road = arguments.get("road")
        if road == "S9":
            return dumps(
                [
                    {
                        "roadName": "S9苏台高速（钱江通道）",
                        "roadGbCode": "S9",
                        "congestionInfoList": [],
                        "trafficControlList": [],
                        "serviceAreaList": [],
                        "exitInfoList": [{"tollName": "周王庙收费站"}],
                    }
                ],
                ensure_ascii=False,
            )
        return dumps([], ensure_ascii=False)


class _UnrelatedNameRetryToolRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        road = arguments.get("road")
        if road == "S9":
            return dumps(
                [
                    {
                        "roadName": "S9苏台高速（钱江通道）",
                        "roadGbCode": "S9",
                        "congestionInfoList": [],
                        "trafficControlList": [],
                        "serviceAreaList": [],
                        "exitInfoList": [{"tollName": "周王庙收费站"}],
                    }
                ],
                ensure_ascii=False,
            )
        return dumps(
            [
                {
                    "roadName": "苏绍高速",
                    "roadGbCode": "S9",
                    "congestionInfoList": [],
                    "trafficControlList": [],
                    "serviceAreaList": [],
                    "exitInfoList": [{"tollName": "前进收费站"}],
                }
            ],
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_traffic_node_uses_llm_inferred_canonical_road() -> None:
    tool_registry = _CapturingToolRegistry()
    node = TrafficNode(tool_registry=tool_registry)

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={
                    "query": "诸暨北收费站温州方向出口堵吗？",
                    "road": "沪昆高速",
                    "road_name": "沪昆高速",
                    "road_code": "G60",
                    "target": "诸暨北收费站温州方向出口",
                    "direction": "温州方向",
                    "toll_station": "诸暨北收费站",
                },
            ),
        }
    )

    assert tool_registry.calls == [
        {
            "tool_name": "live_road_event_query",
            "arguments": {"road": "G60"},
        }
    ]
    query_arguments = result["step_results"]["traffic_1"].raw_result["query_arguments"]
    assert query_arguments["target"] == "诸暨北收费站温州方向出口"
    assert query_arguments["direction"] == "温州方向"
    assert query_arguments["toll_station"] == "诸暨北收费站"


@pytest.mark.asyncio
async def test_traffic_node_queries_road_code_before_name_fallback() -> None:
    tool_registry = _CapturingToolRegistry()
    node = TrafficNode(tool_registry=tool_registry)

    await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={
                    "query": "诸暨北收费站温州方向出口堵吗？",
                    "road": "诸暨北收费站温州方向出口堵吗？",
                    "road_name": "诸永高速",
                    "road_code": "S26",
                    "target": "诸暨北收费站温州方向出口",
                },
            ),
        }
    )

    assert tool_registry.calls[0] == {
        "tool_name": "live_road_event_query",
        "arguments": {"road": "S26"},
    }


@pytest.mark.asyncio
async def test_traffic_node_prefers_single_canonical_road_over_surface_roads_list() -> None:
    tool_registry = _CapturingToolRegistry()
    node = TrafficNode(tool_registry=tool_registry)

    await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={
                    "query": "沪杭高速沪向车道是否畅通",
                    "roads": ["沪杭高速"],
                    "road": "G60",
                    "road_name": "沪昆高速",
                    "road_code": "G60",
                    "target": "沪向车道",
                },
            ),
        }
    )

    assert tool_registry.calls[-1] == {
        "tool_name": "live_road_event_query",
        "arguments": {"road": "G60"},
    }


@pytest.mark.asyncio
async def test_traffic_node_retries_by_road_name_when_code_result_mismatches_name() -> None:
    tool_registry = _MismatchedRoadToolRegistry()
    node = TrafficNode(tool_registry=tool_registry)

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={
                    "query": "Is Shenjiahu congested?",
                    "road": "S32",
                    "road_name": "Shenjiahu Expressway",
                    "road_code": "S32",
                    "target": "Shenjiahu Expressway",
                },
            ),
        }
    )

    assert tool_registry.calls == [
        {"tool_name": "live_road_event_query", "arguments": {"road": "S32"}},
        {
            "tool_name": "live_road_event_query",
            "arguments": {"road": "Shenjiahu Expressway"},
        },
    ]
    query_arguments = result["step_results"]["traffic_1"].raw_result["query_arguments"]
    per_road_results = result["step_results"]["traffic_1"].raw_result["per_road_results"]
    assert query_arguments["queried_roads"] == ["S32"]
    assert per_road_results == [
        {
            "query_road": "Shenjiahu Expressway",
            "api_result": [
                {
                    "roadName": "Shenjiahu Expressway",
                    "roadGbCode": "S12",
                    "congestionInfoList": [],
                    "trafficControlList": [],
                    "serviceAreaList": [],
                    "exitInfoList": [],
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_traffic_node_keeps_code_result_when_name_retry_is_empty() -> None:
    tool_registry = _EmptyNameRetryToolRegistry()
    node = TrafficNode(tool_registry=tool_registry)

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={
                    "query": "周王庙收费站在哪",
                    "road": "S9",
                    "road_name": "苏绍高速",
                    "road_code": "S9",
                    "target": "周王庙收费站",
                    "toll_station": "周王庙收费站",
                },
            ),
        }
    )

    assert tool_registry.calls == [
        {"tool_name": "live_road_event_query", "arguments": {"road": "S9"}},
        {"tool_name": "live_road_event_query", "arguments": {"road": "苏绍高速"}},
    ]
    traffic_result = result["step_results"]["traffic_1"]
    assert traffic_result.normalized_result["result_count"] == 1
    assert traffic_result.normalized_result["matched_road_names"] == ["S9苏台高速（钱江通道）"]
    assert traffic_result.normalized_result["exit_items"] == [
        {
            "road_name": "S9苏台高速（钱江通道）",
            "road_code": "S9",
            "toll_name": "周王庙收费站",
            "toll_id": None,
            "exit_name": None,
            "entrance_status": None,
            "entrance_status_label": None,
            "export_status": None,
            "export_status_label": None,
            "description": None,
        }
    ]


@pytest.mark.asyncio
async def test_traffic_node_keeps_toll_match_when_name_retry_returns_unrelated_result() -> None:
    tool_registry = _UnrelatedNameRetryToolRegistry()
    node = TrafficNode(tool_registry=tool_registry)

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={
                    "query": "周王庙收费站在哪",
                    "road": "S9",
                    "road_name": "苏绍高速",
                    "road_code": "S9",
                    "target": "周王庙收费站",
                    "toll_station": "周王庙收费站",
                },
            ),
        }
    )

    assert tool_registry.calls == [
        {"tool_name": "live_road_event_query", "arguments": {"road": "S9"}},
        {"tool_name": "live_road_event_query", "arguments": {"road": "苏绍高速"}},
    ]
    traffic_result = result["step_results"]["traffic_1"]
    assert traffic_result.normalized_result["matched_road_names"] == ["S9苏台高速（钱江通道）"]
    assert traffic_result.normalized_result["exit_items"] == [
        {
            "road_name": "S9苏台高速（钱江通道）",
            "road_code": "S9",
            "toll_name": "周王庙收费站",
            "toll_id": None,
            "exit_name": None,
            "entrance_status": None,
            "entrance_status_label": None,
            "export_status": None,
            "export_status_label": None,
            "description": None,
        }
    ]
