"""路线规划业务节点模块。
负责把当前路线规划问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问外部路线接口，后续可在此节点内部切换为 HTTP 或 MCP 执行。
"""

from __future__ import annotations

from collections.abc import Iterable
from json import loads

from app.agent.direction_filter import filter_section_events_for_travel_direction
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
                "route_context": self._build_compact_route_context(
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
        route_summaries = RouteNode._build_route_summaries(
            normalized_routes,
            origin=resolved_arguments.arguments.get("origin"),
            destination=resolved_arguments.arguments.get("destination"),
        )
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
        exit_items = [
            item
            for route_summary in route_summaries
            for item in route_summary.get("exit_items", [])
            if isinstance(item, dict)
        ]
        congestion_items = [
            item
            for route_summary in route_summaries
            for item in route_summary.get("congestion_items", [])
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
            "exit_items": exit_items,
            "congestion_items": congestion_items,
            "traffic_control_count": len(traffic_controls),
            "service_area_count": len(service_areas),
            "exit_count": len(exit_items),
            "congestion_count": len(congestion_items),
        }

    @staticmethod
    def _build_route_summaries(
        routes: list[dict[str, object]],
        *,
        origin: object,
        destination: object,
    ) -> list[dict[str, object]]:
        route_summaries: list[dict[str, object]] = []
        for route_index, route in enumerate(routes, start=1):
            sections = RouteNode._extract_sections(route)
            directional_sections = [
                filter_section_events_for_travel_direction(
                    section=section,
                    origin=origin,
                    destination=destination,
                )
                for section in sections
            ]
            traffic_controls = RouteNode._extract_traffic_controls(directional_sections)
            service_areas = RouteNode._extract_service_areas(sections)
            exit_items = RouteNode._extract_exit_items(sections)
            congestion_items = RouteNode._extract_congestion_items(directional_sections)
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
                    "exit_items": exit_items,
                    "congestion_items": congestion_items,
                    "traffic_control_count": len(traffic_controls),
                    "service_area_count": len(service_areas),
                    "exit_count": len(exit_items),
                    "congestion_count": len(congestion_items),
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
                        "control_type": control.get("eventType") or control.get("type"),
                        "start_time": control.get("beginTime") or control.get("startTime"),
                        "end_time": control.get("expectEndTime") or control.get("endTime"),
                        "description": control.get("des") or control.get("description"),
                        "direction_type": control.get("directionType"),
                        "direction_label": RouteNode._normalize_route_direction(
                            control.get("directionType"),
                            control.get("directionName"),
                        ),
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
    def _extract_exit_items(sections: list[dict[str, object]]) -> list[dict[str, object]]:
        exit_items: list[dict[str, object]] = []
        for section in sections:
            road_name = str(section.get("roadName") or "").strip()
            raw_exit_infos = section.get("exitInfos", [])
            if not isinstance(raw_exit_infos, list):
                continue
            for exit_info in raw_exit_infos:
                if not isinstance(exit_info, dict):
                    continue
                exit_items.append(
                    {
                        "road_name": road_name,
                        "toll_name": str(exit_info.get("tollName") or "").strip(),
                        "toll_id": exit_info.get("tollId"),
                        "entrance_status": exit_info.get("entranceStatus"),
                        "entrance_status_label": RouteNode._resolve_station_status_label(
                            exit_info.get("entranceStatus")
                        ),
                        "export_status": exit_info.get("exportStatus"),
                        "export_status_label": RouteNode._resolve_station_status_label(
                            exit_info.get("exportStatus")
                        ),
                    }
                )
        return exit_items

    @staticmethod
    def _extract_congestion_items(sections: list[dict[str, object]]) -> list[dict[str, object]]:
        congestion_items: list[dict[str, object]] = []
        for section in sections:
            road_name = str(section.get("roadName") or "").strip()
            raw_congestions = section.get("trafficCongestions", [])
            if not isinstance(raw_congestions, list):
                continue
            for congestion in raw_congestions:
                if not isinstance(congestion, dict):
                    continue
                begin_milestone = congestion.get("beginMilestone")
                end_milestone = congestion.get("endMilestone")
                congestion_items.append(
                    {
                        "road_name": road_name,
                        "congestion_id": congestion.get("id"),
                        "description": str(
                            congestion.get("des") or congestion.get("description") or ""
                        ).strip(),
                        "begin_time": congestion.get("beginTime"),
                        "control_measures": congestion.get("controlMeasures"),
                        "direction_type": congestion.get("directionType"),
                        "direction_label": RouteNode._normalize_route_direction(
                            congestion.get("directionType"),
                            congestion.get("directionName"),
                        ),
                        "begin_milestone": begin_milestone,
                        "end_milestone": end_milestone,
                        "event_type": congestion.get("eventType"),
                        "sub_event_type": congestion.get("subEventType"),
                        "road_amble_mile": congestion.get("roadAmbleMile"),
                        "road_id": congestion.get("roadId"),
                    }
                )
        return congestion_items

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
    def _normalize_route_direction(direction_type: object, direction_name: object) -> str:
        direction_name_text = str(direction_name or "").strip()
        if direction_name_text:
            return direction_name_text
        direction_type_text = str(direction_type or "").strip()
        if direction_type_text in {"00", "0", "双向"}:
            return "双向"
        stripped_direction = direction_type_text.lstrip("0")
        if stripped_direction == "1":
            return "上行"
        if stripped_direction == "2":
            return "下行"
        return direction_type_text or "未知"

    @staticmethod
    def _normalize_status_code(status_code: object) -> str | None:
        if status_code is None:
            return None
        if isinstance(status_code, bool):
            return str(int(status_code))
        if isinstance(status_code, int):
            return str(status_code)
        if isinstance(status_code, float) and status_code.is_integer():
            return str(int(status_code))

        normalized = str(status_code).strip()
        if not normalized:
            return None
        if normalized.isdigit():
            return str(int(normalized))
        return normalized

    @staticmethod
    def _resolve_station_status_label(status_code: object) -> str | None:
        normalized_status = RouteNode._normalize_status_code(status_code)
        if not normalized_status:
            return None
        if normalized_status == "0":
            return "开启"
        if normalized_status == "10202":
            return "关闭"
        if normalized_status == "10203":
            return "限流"
        if normalized_status == "10204":
            return "分流"
        return normalized_status

    @staticmethod
    def _string_or_placeholder(value: object, placeholder: str) -> str:
        if value is None:
            return placeholder
        text = str(value).strip()
        return text or placeholder

    @staticmethod
    def _format_distance_km(distance_meters: object) -> str:
        if isinstance(distance_meters, (int, float)):
            return f"{distance_meters / 1000:g}"
        return "0"

    @staticmethod
    def _format_duration_hm(duration_minutes: object) -> str:
        if not isinstance(duration_minutes, (int, float)):
            return "0分"
        total_minutes = int(duration_minutes)
        hours, minutes = divmod(total_minutes, 60)
        if hours and minutes:
            return f"{hours}小时{minutes}分"
        if hours:
            return f"{hours}小时"
        return f"{minutes}分"

    @staticmethod
    def _format_toll(toll: object) -> str:
        if isinstance(toll, (int, float)):
            return f"{toll:g}"
        return "0"

    @staticmethod
    def _format_milestone(value: object) -> str:
        return RouteNode._string_or_placeholder(value, "未知")

    @staticmethod
    def _extract_route_service_items(sections: list[dict[str, object]]) -> list[dict[str, str]]:
        service_items: list[dict[str, str]] = []
        for section in sections:
            raw_service_areas = section.get("serviceAreas", [])
            if not isinstance(raw_service_areas, list):
                continue
            for service_area in raw_service_areas:
                if not isinstance(service_area, dict):
                    continue
                service_name = str(service_area.get("serviceName") or "").strip()
                if not service_name:
                    continue
                service_items.append(
                    {
                        "service_name": service_name,
                        "direction": RouteNode._normalize_route_direction(
                            service_area.get("directionType"),
                            service_area.get("directionName"),
                        ),
                    }
                )
        return service_items

    @staticmethod
    def _extract_route_control_items(sections: list[dict[str, object]]) -> list[dict[str, str]]:
        control_items: list[dict[str, str]] = []
        for section in sections:
            raw_controls = section.get("trafficControls", [])
            if not isinstance(raw_controls, list):
                continue
            for control in raw_controls:
                if not isinstance(control, dict):
                    continue
                control_items.append(
                    {
                        "begin_milestone": RouteNode._string_or_placeholder(
                            control.get("beginMilestone"),
                            "未知",
                        ),
                        "end_milestone": RouteNode._string_or_placeholder(
                            control.get("endMilestone"),
                            "未知",
                        ),
                        "direction": RouteNode._normalize_route_direction(
                            control.get("directionType"),
                            control.get("directionName"),
                        ),
                        "description": RouteNode._string_or_placeholder(
                            control.get("des") or control.get("description"),
                            "暂无描述",
                        ),
                        "begin_time": RouteNode._string_or_placeholder(
                            control.get("beginTime"),
                            "未知",
                        ),
                        "expected_end_time": RouteNode._string_or_placeholder(
                            control.get("expectedEndTime") or control.get("endTime"),
                            "未知",
                        ),
                        "control_measures": RouteNode._string_or_placeholder(
                            control.get("controlMeasures"),
                            "暂无",
                        ),
                    }
                )
        return control_items

    @staticmethod
    def _build_compact_route_block(
        *,
        route: dict[str, object],
        route_index: int,
        origin: object,
        destination: object,
    ) -> str:
        tags = RouteNode._deduplicate_strings(route.get("tags", []))
        tag_text = "、".join(tags) if tags else "未标注"
        distance_km = RouteNode._format_distance_km(route.get("distance"))
        duration_hm = RouteNode._format_duration_hm(route.get("duration"))
        toll_text = RouteNode._format_toll(route.get("toll"))

        sections = RouteNode._extract_sections(route)
        directional_sections = [
            filter_section_events_for_travel_direction(
                section=section,
                origin=origin,
                destination=destination,
            )
            for section in sections
        ]
        road_names = RouteNode._extract_road_names(sections)
        service_items = RouteNode._extract_route_service_items(sections)
        control_items = RouteNode._extract_route_control_items(directional_sections)
        exit_items = RouteNode._extract_exit_items(sections)
        congestion_items = RouteNode._extract_congestion_items(directional_sections)

        lines = [
            f"方案 {route_index} [{tag_text}]：路线共{distance_km}km | 预计耗时{duration_hm} | 费用过路费{toll_text}元",
            "途经路段：" + (" → ".join(road_names) if road_names else "暂无"),
            "途经服务区：",
        ]

        if service_items:
            lines.append("、".join(f"{item['service_name']}（{item['direction']}）" for item in service_items) + "、")
        else:
            lines.append("暂无")

        lines.append("沿途交通管制：")
        if control_items:
            for item in control_items:
                lines.append(
                    "  - "
                    f"K{item['begin_milestone']}-K{item['end_milestone']}"
                    f"（{item['direction']}）：{item['description']} | "
                    f"{item['begin_time']}-{item['expected_end_time']} | "
                    f"管制措施：{item['control_measures']}"
                )
        else:
            lines.append("  - 暂无")

        lines.append("沿途收费站/出口：")
        if exit_items:
            for item in exit_items:
                lines.append(
                    "  - "
                    f"{item['toll_name'] or '未知收费站'}"
                    f"（入口{item['entrance_status_label'] or '未知'} / "
                    f"出口{item['export_status_label'] or '未知'}）"
                )
        else:
            lines.append("  - 暂无")

        lines.append("沿途拥堵信息：")
        if congestion_items:
            for item in congestion_items:
                lines.append(
                    "  - "
                    f"K{RouteNode._format_milestone(item['begin_milestone'])}"
                    f"-K{RouteNode._format_milestone(item['end_milestone'])}"
                    f"（{item['direction_label']}）：{item['description'] or '暂无描述'} | "
                    f"开始时间：{RouteNode._string_or_placeholder(item['begin_time'], '未知')} | "
                    f"管制措施：{RouteNode._string_or_placeholder(item['control_measures'], '暂无')}"
                )
        else:
            lines.append("  - 暂无")

        return "\n".join(lines)

    @staticmethod
    def _build_compact_route_context(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: dict[str, object],
    ) -> str:
        routes = response_payload.get("routes", [])
        normalized_routes = (
            [route for route in routes if isinstance(route, dict)]
            if isinstance(routes, list)
            else []
        )
        if not normalized_routes:
            return ""

        route_count = response_payload.get("routesCount")
        if route_count is None:
            route_count = len(normalized_routes)

        lines = [
            ROUTE_CONTEXT_PROMPT_PREFIX.rstrip(),
            (
                f"查询参数：起点：{RouteNode._string_or_placeholder(resolved_arguments.arguments.get('origin'), '未知')}，"
                f"终点：{RouteNode._string_or_placeholder(resolved_arguments.arguments.get('destination'), '未知')},"
                f"共查询路线方案（共 {route_count} 条）："
            ),
        ]

        for route_index, route in enumerate(normalized_routes, start=1):
            lines.append(
                RouteNode._build_compact_route_block(
                    route=route,
                    route_index=route_index,
                    origin=resolved_arguments.arguments.get("origin"),
                    destination=resolved_arguments.arguments.get("destination"),
                )
            )

        return "\n".join(lines)

    @staticmethod
    def _build_success_summary(response_payload: dict[str, object]) -> str:
        """生成路线查询成功摘要。"""

        routes = response_payload.get("routes", [])
        route_count = response_payload.get("routesCount")
        if route_count is None and isinstance(routes, list):
            route_count = len(routes)
        return f"路线查询成功，命中 {route_count or 0} 条路线方案。"
