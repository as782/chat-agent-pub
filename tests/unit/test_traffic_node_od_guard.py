from __future__ import annotations

from json import dumps

from app.agent.nodes.traffic_node import TrafficNode
from app.agent.state import ExecutionPlan, ResolvedArguments


class _CapturingToolRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        return dumps([], ensure_ascii=False)


async def test_traffic_node_does_not_query_raw_od_text_when_route_roads_are_missing() -> None:
    tool_registry = _CapturingToolRegistry()
    node = TrafficNode(tool_registry=tool_registry)

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="multi_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={
                    "query": "宁波到杭州路况如何？",
                    "target": "宁波至杭州全程",
                    "query_intent": "route_based_traffic",
                },
            ),
        }
    )

    assert tool_registry.calls == []
    query_arguments = result["step_results"]["traffic_1"].raw_result["query_arguments"]
    assert query_arguments["road"] == ""
    assert query_arguments["queried_roads"] == []
    assert query_arguments["target"] == "宁波至杭州全程"
