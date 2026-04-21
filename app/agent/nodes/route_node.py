"""路线规划业务节点模块。
负责把当前路线规划问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问外部路线接口，后续可在此节点内部切换为 HTTP 或 MCP 执行。
"""

from __future__ import annotations

from collections.abc import Iterable
from json import dumps, loads

from app.agent.prompts import ROUTE_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
from app.agent.state import (
    AgentState,
    ExecutorResult,
    ResolvedArguments,
    merge_step_result,
    resolve_active_execution_step_id,
    resolve_step_arguments,
)
from app.core.exceptions import UpstreamServiceException
from app.tools.registry import ToolRegistry


class RouteNode:
    """LangGraph 路线规划业务节点。"""

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or ToolRegistry()

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行路线查询工具并生成路线规划上下文。"""

        step_id = resolve_active_execution_step_id(
            state,
            executor="route",
            default_step_id="route_1",
        )
        resolved_arguments = resolve_step_arguments(state, step_id=step_id, executor="route")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"route_context": None}
        query_arguments = self._build_tool_arguments(resolved_arguments)
        try:
            tool_output = await self._tool_registry.execute_named_tool(
                tool_name="live_driving_query",
                arguments=query_arguments,
            )
            response_payload = self._parse_tool_output(tool_output)
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="route",
                is_success=True,
                raw_result={
                    "query_arguments": dict(query_arguments),
                    "api_result": response_payload,
                },
                normalized_result=self._build_normalized_result(
                    resolved_arguments=resolved_arguments,
                    response_payload=response_payload,
                ),
                summary=self._build_success_summary(response_payload),
            )
            return {
                "route_context": self._build_route_context(
                    resolved_arguments=resolved_arguments,
                    response_payload=response_payload,
                ),
                **merge_step_result(state, result=executor_result),
            }
        except UpstreamServiceException as exception:
            raise UpstreamServiceException(
                UPSTREAM_SERVICE_ERROR_REPLY,
                error_code=exception.error_code,
                status_code=exception.status_code,
                details=exception.details,
            ) from exception

    @staticmethod
    def _build_tool_arguments(resolved_arguments: ResolvedArguments) -> dict[str, object]:
        """把结构化参数转换为路线查询工具参数。"""

        return {
            "start": str(resolved_arguments.arguments.get("origin") or ""),
            "end": str(resolved_arguments.arguments.get("destination") or ""),
        }

    @staticmethod
    def _parse_tool_output(tool_output: str) -> dict[str, object]:
        """解析路线工具返回的 JSON 字符串。"""

        response_payload = loads(tool_output)
        return response_payload if isinstance(response_payload, dict) else {}

    @staticmethod
    def _build_normalized_result(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: dict[str, object],
    ) -> dict[str, object]:
        """提取路线查询结果中的完整业务字段。"""

        routes = response_payload.get("routes", [])
        normalized_routes = (
            [route for route in routes if isinstance(route, dict)]
            if isinstance(routes, list)
            else []
        )
        first_route = normalized_routes[0] if normalized_routes else {}
        route_summaries = RouteNode._build_route_summaries(normalized_routes)
        traffic_controls = [
            item
            for route_summary in route_summaries
            for item in route_summary.get("traffic_controls", [])
            if isinstance(item, dict)
        ]
        service_areas = [
            item
            for route_summary in route_summaries
            for item in route_summary.get("service_areas", [])
            if isinstance(item, dict)
        ]
        road_names = RouteNode._deduplicate_strings(
            road_name
            for route_summary in route_summaries
            for road_name in route_summary.get("road_names", [])
        )
        service_area_names = [
            item["service_name"] for item in service_areas if item.get("service_name")
        ]
        return {
            "origin": resolved_arguments.arguments.get("origin"),
            "destination": resolved_arguments.arguments.get("destination"),
            "travel_mode": resolved_arguments.arguments.get("travel_mode"),
            "routes_count": response_payload.get("routesCount")
            if response_payload.get("routesCount") is not None
            else len(normalized_routes),
            "first_route_distance": first_route.get("distance"),
            "first_route_duration": first_route.get("duration"),
            "first_route_toll": first_route.get("toll"),
            "road_names": road_names,
            "route_summaries": route_summaries,
            "service_area_names": RouteNode._deduplicate_strings(service_area_names),
            "traffic_controls": traffic_controls,
            "service_areas": service_areas,
            "traffic_control_count": len(traffic_controls),
            "service_area_count": len(service_areas),
        }

    @staticmethod
    def _build_route_summaries(routes: list[dict[str, object]]) -> list[dict[str, object]]:
        route_summaries: list[dict[str, object]] = []
        for route_index, route in enumerate(routes, start=1):
            sections = RouteNode._extract_sections(route)
            traffic_controls = RouteNode._extract_traffic_controls(sections)
            service_areas = RouteNode._extract_service_areas(sections)
            road_names = RouteNode._extract_road_names(sections)
            route_summaries.append(
                {
                    "route_index": route_index,
                    "distance": route.get("distance"),
                    "duration": route.get("duration"),
                    "toll": route.get("toll"),
                    "section_count": len(sections),
                    "road_names": road_names,
                    "traffic_controls": traffic_controls,
                    "service_areas": service_areas,
                    "traffic_control_count": len(traffic_controls),
                    "service_area_count": len(service_areas),
                }
            )
        return route_summaries

    @staticmethod
    def _extract_sections(first_route: dict[str, object]) -> list[dict[str, object]]:
        sections = first_route.get("sections", [])
        if not isinstance(sections, list):
            return []
        return [section for section in sections if isinstance(section, dict)]

    @staticmethod
    def _extract_road_names(sections: list[dict[str, object]]) -> list[str]:
        """
        提取路线中的道路名称。
        """
        return RouteNode._deduplicate_strings(
            str(section.get("roadName") or "").strip()
            for section in sections
        )

    @staticmethod
    def _extract_traffic_controls(sections: list[dict[str, object]]) -> list[dict[str, object]]:
        traffic_controls: list[dict[str, object]] = []
        for section in sections:
            road_name = str(section.get("roadName") or "").strip()
            raw_controls = section.get("trafficControls", [])
            if not isinstance(raw_controls, list):
                continue
            for control in raw_controls:
                if not isinstance(control, dict):
                    continue
                traffic_controls.append(
                    {
                        "road_name": road_name,
                        "control_id": control.get("id"),
                        "control_name": control.get("name") or control.get("controlName"),
                        "control_type": control.get("eventType")  or control.get("type") ,
                        "start_time":  control.get("beginTime") or control.get("startTime"),
                        "end_time":control.get("expectEndTime") or control.get("endTime"),
                        "description": control.get("des") or control.get("description"),
                        
                    }
                )
        return traffic_controls

    @staticmethod
    def _extract_service_areas(sections: list[dict[str, object]]) -> list[dict[str, object]]:
        service_areas: list[dict[str, object]] = []
        for section in sections:
            road_name = str(section.get("roadName") or "").strip()
            raw_service_areas = section.get("serviceAreas", [])
            if not isinstance(raw_service_areas, list):
                continue
            for service_area in raw_service_areas:
                if not isinstance(service_area, dict):
                    continue
                service_areas.append(
                    {
                        "road_name": road_name,
                        "service_name": str(service_area.get("serviceName") or "").strip(),
                        "service_id": service_area.get("id"),
                        "direction": service_area.get("direction"),
                        "distance": service_area.get("distance"),
                        "status_tag": service_area.get("statusTag"),
                    }
                )
        return service_areas

    @staticmethod
    def _deduplicate_strings(values: Iterable[object]) -> list[str]:
        """
            去重字符串
        """
        seen: set[str] = set()
        ordered_values: list[str] = []
        for raw_value in values:
            value = str(raw_value).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered_values.append(value)
        return ordered_values

    @staticmethod
    def _build_success_summary(response_payload: dict[str, object]) -> str:
        """生成路线查询成功摘要。"""

        routes = response_payload.get("routes", [])
        route_count = response_payload.get("routesCount")
        if route_count is None and isinstance(routes, list):
            route_count = len(routes)
        return f"路线查询成功，命中 {route_count or 0} 条路线方案。"

    @staticmethod
    def _build_route_context(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: dict[str, object],
    ) -> str:
        """把结构化参数和接口返回拼成路线类 system 上下文。"""

        return "\n".join(
            [
                ROUTE_CONTEXT_PROMPT_PREFIX,
                dumps(
                    {
                        "query_arguments": dict(resolved_arguments.arguments),
                        "api_result": response_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
