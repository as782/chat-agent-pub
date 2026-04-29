"""OD resolver tests."""

from __future__ import annotations

import json

import pytest

from app.agent.nodes.argument_node import ArgumentNode
from app.agent.nodes.route_node import RouteNode
from app.agent.od import resolve_od
from app.agent.state import ExecutionPlan, ExecutionStep, ResolvedArguments


def test_od_resolver_cleans_current_location_service_area_query() -> None:
    resolution = resolve_od("用户问题：\n我现在正在益农服务区往杭州去，我还有多少时间能到杭州？")

    assert resolution.origin == "益农服务区"
    assert resolution.destination == "杭州"
    assert resolution.origin_match_type == "facility_service_area"


def test_od_resolver_normalizes_toll_station_without_service_area_cross_match() -> None:
    resolution = resolve_od("枫林主收费站往杭州去，我还有多少时间能到杭州？")

    assert resolution.origin == "枫林主（温向）收费站"
    assert resolution.destination == "杭州"
    assert resolution.origin_match_type == "facility_toll_station"


def test_od_resolver_prefers_repeated_arrival_destination_anchor() -> None:
    resolution = resolve_od("枫林主收费站往杭州区去，我还有多少时间能到杭州？")

    assert resolution.origin == "枫林主（温向）收费站"
    assert resolution.destination == "杭州"


def test_od_resolver_keeps_two_toll_station_endpoint_names() -> None:
    resolution = resolve_od("金华收费站到上海收费站路线规划及路况")

    assert resolution.origin == "金华收费站"
    assert resolution.destination == "上海收费站"
    assert resolution.source == "local_od_resolver:place_index"


def test_od_resolver_uses_region_catalog_to_trim_city_destination_noise() -> None:
    resolution = resolve_od("金华到上海正么样")

    assert resolution.origin == "金华"
    assert resolution.destination == "上海"
    assert resolution.source == "local_od_resolver:place_index"


def test_od_resolver_combines_facility_origin_and_region_destination() -> None:
    resolution = resolve_od("金华收费站到上海路线及路况")

    assert resolution.origin == "金华收费站"
    assert resolution.destination == "上海"
    assert resolution.source == "local_od_resolver:place_index"


@pytest.mark.parametrize(
    ("message", "origin", "destination"),
    [
        ("杭州到金华怎么走", "杭州", "金华"),
        ("苍南去玉环堵不堵", "苍南", "玉环"),
        ("从苏州到杭州要多久", "苏州", "杭州"),
        ("金华到上海啥时候能到", "金华", "上海"),
        ("我在张三工业园往杭州去", "张三工业园", "杭州"),
    ],
)
def test_od_resolver_extracts_clean_structured_text_endpoints(
    message: str,
    origin: str,
    destination: str,
) -> None:
    resolution = resolve_od(message)

    assert resolution.origin == origin
    assert resolution.destination == destination


@pytest.mark.asyncio
async def test_argument_node_keeps_clean_resolver_od_over_dirty_planner_metadata() -> None:
    node = ArgumentNode()

    latest_user_message = "用户问题：\n我现在正在益农服务区往杭州去，我还有多少时间能到杭州？"
    result = await node.run(
        {
            "latest_user_message": latest_user_message,
            "primary_category": "route_planning",
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
                steps=[
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="查询路线",
                        metadata={
                            "origin": "我现在正在益农服务区",
                            "destination": "杭州去",
                            "travel_mode": "auto",
                        },
                    )
                ],
            ),
        }
    )

    route_arguments = result["step_arguments"]["route_1"].arguments
    assert route_arguments["origin"] == "益农服务区"
    assert route_arguments["destination"] == "杭州"
    assert result["need_clarification"] is False


@pytest.mark.asyncio
async def test_argument_node_uses_shorter_clean_planner_destination_prefix() -> None:
    node = ArgumentNode()

    result = await node.run(
        {
            "latest_user_message": "金华收费站到上海正么样",
            "primary_category": "route_planning",
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
                steps=[
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="查询路线",
                        metadata={
                            "origin": "金华收费站",
                            "destination": "上海",
                            "travel_mode": "auto",
                        },
                    )
                ],
            ),
        }
    )

    route_arguments = result["step_arguments"]["route_1"].arguments
    assert route_arguments["origin"] == "金华收费站"
    assert route_arguments["destination"] == "上海"
    assert result["need_clarification"] is False


class _CapturingRouteToolRegistry:
    def __init__(self) -> None:
        self.arguments: dict[str, object] | None = None

    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        self.arguments = dict(arguments)
        return json.dumps({"routes": [], "routesCount": 0}, ensure_ascii=False)


class _FailingRouteToolRegistry:
    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        raise AssertionError("route tool should not be called with invalid OD arguments")


@pytest.mark.asyncio
async def test_route_node_repairs_dirty_arguments_before_tool_call() -> None:
    registry = _CapturingRouteToolRegistry()
    node = RouteNode(tool_registry=registry)

    latest_user_message = "用户问题：\n我现在正在益农服务区往杭州去，我还有多少时间能到杭州？"
    await node.run(
        {
            "latest_user_message": latest_user_message,
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={"origin": "我现在正在益农服务区", "destination": "杭州去"},
            ),
        }
    )

    assert registry.arguments == {"start": "益农服务区", "end": "杭州"}


@pytest.mark.asyncio
async def test_route_node_blocks_invalid_arguments_when_repair_fails() -> None:
    node = RouteNode(tool_registry=_FailingRouteToolRegistry())

    result = await node.run(
        {
            "latest_user_message": "还有多少时间能到？",
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={"origin": "我现在正在", "destination": "杭州去"},
            ),
        }
    )

    route_result = result["step_results"]["route_1"]
    assert route_result.is_success is False
    assert route_result.raw_result["validation_warnings"]
