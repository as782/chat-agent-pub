from __future__ import annotations

from json import dumps

import pytest

from app.agent.nodes.traffic_node import TrafficNode
from app.agent.planner import PlannerService
from app.agent.road_inference import normalize_traffic_road_fields
from app.agent.state import ExecutionPlan, ResolvedArguments


class _CapturingToolRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        return dumps(
            [
                {
                    "roadName": "宁波绕城高速",
                    "roadGbCode": "G1503",
                    "congestionInfoList": [],
                    "trafficControlList": [],
                    "serviceAreaList": [],
                    "exitInfoList": [{"tollName": "宁波东收费站"}],
                }
            ],
            ensure_ascii=False,
        )


def test_normalize_traffic_road_fields_splits_mixed_identifier() -> None:
    normalized = normalize_traffic_road_fields(
        road="G1503 宁波绕城高速",
        road_name="G1503 宁波绕城高速",
        road_code="g1503",
        prefer="name",
    )

    assert normalized == {
        "road": "宁波绕城高速",
        "road_name": "宁波绕城高速",
        "road_code": "G1503",
    }


def test_planner_normalizes_mixed_traffic_road_identifier_fields() -> None:
    planner = PlannerService()

    metadata = planner._enrich_step_metadata(
        executor="traffic",
        metadata={
            "query": "宁波东收费站堵车吗？",
            "road": "G1503 宁波绕城高速",
            "road_name": "G1503 宁波绕城高速",
            "road_code": "g1503",
            "roads": ["G1503 宁波绕城高速"],
            "target": "宁波东收费站",
            "query_intent": "traffic_status",
        },
        latest_user_message="宁波东收费站堵车吗？",
        primary_category="traffic_status",
    )

    assert metadata["road"] == "宁波绕城高速"
    assert metadata["road_name"] == "宁波绕城高速"
    assert metadata["road_code"] == "G1503"
    assert metadata["roads"] == ["宁波绕城高速"]


@pytest.mark.asyncio
async def test_traffic_node_normalizes_mixed_road_identifier_before_query() -> None:
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
                    "query": "宁波东收费站堵车吗？",
                    "road": "G1503 宁波绕城高速",
                    "road_name": "G1503 宁波绕城高速",
                    "road_code": "g1503",
                    "target": "宁波东收费站",
                },
            ),
        }
    )

    assert tool_registry.calls == [
        {
            "tool_name": "live_road_event_query",
            "arguments": {"road": "宁波绕城高速"},
        }
    ]
    query_arguments = result["step_results"]["traffic_1"].raw_result["query_arguments"]
    assert query_arguments["road"] == "宁波绕城高速"
    assert query_arguments["road_name"] == "宁波绕城高速"
    assert query_arguments["road_code"] == "G1503"
