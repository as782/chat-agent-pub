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


def test_planner_splits_mixed_road_identifier_into_name_and_code() -> None:
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

    assert metadata["road"] == "G92"
    assert metadata["road_name"] == "杭州湾跨海大桥连接线"
    assert metadata["road_code"] == "G92"
    assert metadata["toll_station"] == "宁波东收费站"


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

    assert "不要假设本地还有额外映射表帮你兜底" in combined_prompt
    assert "诸暨北收费站温州方向出口堵吗" in combined_prompt
    assert "沪杭高速沪向车道全部畅通吗" in combined_prompt
    assert "单路场景至少必须填写 road" in combined_prompt
    assert "road_name、road_code 也必须一起补齐" in combined_prompt
    assert "默认优先填写纯道路编号" in combined_prompt
    assert "不能写成 “G92杭州湾跨海大桥连接线”" in combined_prompt


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
async def test_traffic_node_prefers_road_code_over_road_name_and_raw_query_text() -> None:
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

    assert tool_registry.calls[-1] == {
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
