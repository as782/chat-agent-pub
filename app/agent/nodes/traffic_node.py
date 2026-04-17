"""路况业务节点模块。

负责把当前路况问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问实时路况接口。
"""

from __future__ import annotations

from collections.abc import Iterable
from json import dumps, loads

from app.agent.prompts import TRAFFIC_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
from app.agent.road_inference import (
    normalize_road_query_list,
    normalize_traffic_road_fields,
)
from app.agent.state import (
    AgentState,
    ExecutorResult,
    ResolvedArguments,
    get_execution_step,
    merge_step_result,
    resolve_active_execution_step_id,
    resolve_step_arguments,
)
from app.core.exceptions import UpstreamServiceException
from app.core.logger import get_logger
from app.tools.registry import ToolRegistry

LOGGER = get_logger(__name__)


class TrafficNode:
    """LangGraph 路况业务节点。"""

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or ToolRegistry()

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行路况查询工具并生成路况业务上下文。"""

        step_id = resolve_active_execution_step_id(
            state,
            executor="traffic",
            default_step_id="traffic_1",
        )
        resolved_arguments = resolve_step_arguments(state, step_id=step_id, executor="traffic")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"traffic_context": None}
        query_arguments = self._build_tool_arguments(
            state=state,
            step_id=step_id,
            resolved_arguments=resolved_arguments,
        )
        try:
            per_road_results = await self._query_roads(query_arguments)
            response_payload = self._merge_road_payloads(per_road_results)
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="traffic",
                is_success=True,
                raw_result={
                    "query_arguments": dict(query_arguments),
                    "api_result": response_payload,
                    "per_road_results": per_road_results,
                },
                normalized_result=self._build_normalized_result(
                    query_arguments=query_arguments,
                    response_payload=response_payload,
                    per_road_results=per_road_results,
                ),
                summary=self._build_success_summary(
                    response_payload=response_payload,
                    per_road_results=per_road_results,
                ),
            )
            return {
                "traffic_context": self._build_traffic_context(
                    query_arguments=query_arguments,
                    response_payload=response_payload,
                    per_road_results=per_road_results,
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

    def _build_tool_arguments(
        self,
        *,
        state: AgentState,
        step_id: str,
        resolved_arguments: ResolvedArguments,
    ) -> dict[str, object]:
        """把结构化参数转换为路况查询工具参数。"""

        route_summaries = self._resolve_route_summaries(state=state, step_id=step_id)
        queried_roads = self._resolve_queried_roads(
            state=state,
            step_id=step_id,
            resolved_arguments=resolved_arguments,
        )
        road = queried_roads[0] if queried_roads else ""
        query_arguments: dict[str, object] = {
            "road": road,
            "queried_roads": queried_roads,
            "route_summaries": route_summaries,
        }
        for key in ("target", "direction", "toll_station", "road_name", "road_code", "query"):
            value = resolved_arguments.arguments.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            query_arguments[key] = value

        query_arguments.update(
            normalize_traffic_road_fields(
                road=query_arguments.get("road"),
                road_name=query_arguments.get("road_name"),
                road_code=query_arguments.get("road_code"),
                prefer="name",
            )
        )
        normalized_queried_roads = normalize_road_query_list(queried_roads, prefer="name")
        query_arguments["queried_roads"] = normalized_queried_roads
        query_arguments["road"] = normalized_queried_roads[0] if normalized_queried_roads else str(
            query_arguments.get("road") or ""
        )

        if (
            str(resolved_arguments.arguments.get("query_intent") or "").strip() == "route_based_traffic"
            and not normalized_queried_roads
        ):
            LOGGER.warning(
                "Traffic node did not resolve concrete roads for route-based traffic query: step_id=%s target=%s query=%s",
                step_id,
                query_arguments.get("target"),
                query_arguments.get("query"),
            )
        return query_arguments

    def _resolve_route_summaries(
        self,
        *,
        state: AgentState,
        step_id: str,
    ) -> list[dict[str, object]]:
        current_step = get_execution_step(state, step_id=step_id)
        if current_step is None:
            return []

        step_results = state.get("step_results", {})
        if not isinstance(step_results, dict):
            return []

        for dependency in current_step.depends_on:
            dependency_result = step_results.get(dependency)
            if not isinstance(dependency_result, ExecutorResult):
                continue
            if dependency_result.executor != "route":
                continue
            raw_route_summaries = dependency_result.normalized_result.get("route_summaries")
            if isinstance(raw_route_summaries, list):
                return [item for item in raw_route_summaries if isinstance(item, dict)]
        return []

    async def _query_roads(
        self,
        query_arguments: dict[str, object],
    ) -> list[dict[str, object]]:
        queried_roads = query_arguments.get("queried_roads")
        normalized_roads = (
            self._deduplicate_strings(queried_roads)
            if isinstance(queried_roads, list)
            else []
        )

        road_results: list[dict[str, object]] = []
        for road in normalized_roads:
            tool_output = await self._tool_registry.execute_named_tool(
                tool_name="live_road_event_query",
                arguments={"road": road},
            )
            road_results.append(
                {
                    "query_road": road,
                    "api_result": self._parse_tool_output(tool_output),
                }
            )
        return road_results

    def _resolve_queried_roads(
        self,
        *,
        state: AgentState,
        step_id: str,
        resolved_arguments: ResolvedArguments,
    ) -> list[str]:
        current_step = get_execution_step(state, step_id=step_id)
        if current_step is not None:
            step_results = state.get("step_results", {})
            if isinstance(step_results, dict):
                for dependency in current_step.depends_on:
                    dependency_result = step_results.get(dependency)
                    if not isinstance(dependency_result, ExecutorResult):
                        continue
                    if dependency_result.executor != "route":
                        continue
                    route_summaries = dependency_result.normalized_result.get("route_summaries")
                    if isinstance(route_summaries, list):
                        normalized_road_names = normalize_road_query_list(
                            [
                                str(road_name)
                                for route_summary in route_summaries
                                if isinstance(route_summary, dict)
                                for road_name in route_summary.get("road_names", [])
                            ],
                            prefer="name",
                        )
                        if normalized_road_names:
                            return normalized_road_names
                    road_names = dependency_result.normalized_result.get("road_names")
                    if isinstance(road_names, list):
                        normalized_road_names = normalize_road_query_list(
                            [str(road_name) for road_name in road_names],
                            prefer="name",
                        )
                        if normalized_road_names:
                            return normalized_road_names

        resolved_roads = resolved_arguments.arguments.get("roads")
        if isinstance(resolved_roads, list):
            normalized_road_names = normalize_road_query_list(
                [str(road_name) for road_name in resolved_roads],
                prefer="name",
            )
            if normalized_road_names:
                return normalized_road_names

        fallback_road = self._resolve_single_road_argument(resolved_arguments)
        return [fallback_road] if fallback_road else []

    @staticmethod
    def _resolve_single_road_argument(resolved_arguments: ResolvedArguments) -> str:
        """Prefer structured LLM/planner road fields before falling back to raw text."""

        arguments = resolved_arguments.arguments
        normalized_road_fields = normalize_traffic_road_fields(
            road=arguments.get("road"),
            road_name=arguments.get("road_name"),
            road_code=arguments.get("road_code"),
            prefer="name",
        )
        query_intent = str(arguments.get("query_intent") or "").strip()
        if query_intent == "route_based_traffic":
            for key in ("road", "road_name", "road_code"):
                value = str(normalized_road_fields.get(key) or "").strip()
                if value:
                    return value
            return ""

        for key in ("road", "road_name", "road_code"):
            value = str(normalized_road_fields.get(key) or "").strip()
            if value:
                return value
        for key in ("target", "query"):
            value = str(arguments.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _merge_road_payloads(per_road_results: list[dict[str, object]]) -> list[dict[str, object]]:
        merged_payloads: list[dict[str, object]] = []
        for road_result in per_road_results:
            api_result = road_result.get("api_result")
            if not isinstance(api_result, list):
                continue
            merged_payloads.extend(item for item in api_result if isinstance(item, dict))
        return merged_payloads

    @staticmethod
    def _parse_tool_output(tool_output: str) -> list[dict[str, object]]:
        """解析路况工具返回的 JSON 字符串。"""

        response_payload = loads(tool_output)
        if isinstance(response_payload, list):
            return [item for item in response_payload if isinstance(item, dict)]
        return []

    @staticmethod
    def _build_normalized_result(
        *,
        query_arguments: dict[str, object],
        response_payload: list[dict[str, object]],
        per_road_results: list[dict[str, object]],
    ) -> dict[str, object]:
        """提取路况查询结果中的完整业务字段。"""

        first_road = response_payload[0] if response_payload else {}
        matched_road_names = TrafficNode._deduplicate_strings(
            str(item.get("roadName") or "").strip() for item in response_payload
        )
        congestion_items = TrafficNode._extract_congestion_items(response_payload)
        traffic_control_items = TrafficNode._extract_traffic_control_items(response_payload)
        service_area_items = TrafficNode._extract_service_area_items(response_payload)
        exit_items = TrafficNode._extract_exit_items(response_payload)
        event_items = TrafficNode._build_event_items(
            congestion_items=congestion_items,
            traffic_control_items=traffic_control_items,
            service_area_items=service_area_items,
            exit_items=exit_items,
        )
        queried_roads = query_arguments.get("queried_roads")
        road_summaries = TrafficNode._build_road_summaries(per_road_results)
        route_summaries = TrafficNode._build_route_summaries(
            query_arguments=query_arguments,
            per_road_results=per_road_results,
        )
        return {
            "road": query_arguments.get("road"),
            "road_name": query_arguments.get("road_name") or first_road.get("roadName"),
            "road_code": query_arguments.get("road_code"),
            "target": query_arguments.get("target"),
            "direction": query_arguments.get("direction"),
            "toll_station": query_arguments.get("toll_station"),
            "queried_roads": queried_roads if isinstance(queried_roads, list) else [],
            "requested_road_count": len(queried_roads) if isinstance(queried_roads, list) else 0,
            "matched_road_names": matched_road_names,
            "matched_road_count": len(matched_road_names),
            "result_count": len(response_payload),
            "has_congestion": bool(congestion_items),
            "congestion_count": len(congestion_items),
            "congestion_items": congestion_items,
            "has_control": bool(traffic_control_items),
            "traffic_control_count": len(traffic_control_items),
            "traffic_control_items": traffic_control_items,
            "service_area_count": len(service_area_items),
            "service_area_items": service_area_items,
            "exit_count": len(exit_items),
            "exit_items": exit_items,
            "event_count": len(event_items),
            "event_items": event_items,
            "road_summaries": road_summaries,
            "route_summaries": route_summaries,
            "route_count": len(route_summaries),
        }

    @staticmethod
    def _build_success_summary(
        *,
        response_payload: list[dict[str, object]],
        per_road_results: list[dict[str, object]],
    ) -> str:
        """生成路况查询成功摘要。"""

        if len(per_road_results) > 1:
            return (
                f"多道路路况查询成功，查询 {len(per_road_results)} 条道路，"
                f"命中 {len(response_payload)} 条道路结果。"
            )
        return f"路况查询成功，命中 {len(response_payload)} 条道路结果。"

    @staticmethod
    def _build_traffic_context(
        *,
        query_arguments: dict[str, object],
        response_payload: list[dict[str, object]],
        per_road_results: list[dict[str, object]],
    ) -> str:
        """把结构化参数和接口返回转为路况类 system 上下文。"""

        return "\n".join(
            [
                TRAFFIC_CONTEXT_PROMPT_PREFIX,
                dumps(
                    {
                        "query_arguments": dict(query_arguments),
                        "api_result": response_payload,
                        "per_road_results": per_road_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )

    @staticmethod
    def _build_road_summaries(per_road_results: list[dict[str, object]]) -> list[dict[str, object]]:
        road_summaries: list[dict[str, object]] = []
        for road_result in per_road_results:
            query_road = str(road_result.get("query_road") or "").strip()
            api_result = road_result.get("api_result")
            payload = (
                [item for item in api_result if isinstance(item, dict)]
                if isinstance(api_result, list)
                else []
            )
            matched_road_names = TrafficNode._deduplicate_strings(
                str(item.get("roadName") or "").strip() for item in payload
            )
            congestion_items = TrafficNode._extract_congestion_items(payload)
            traffic_control_items = TrafficNode._extract_traffic_control_items(payload)
            service_area_items = TrafficNode._extract_service_area_items(payload)
            exit_items = TrafficNode._extract_exit_items(payload)
            event_items = TrafficNode._build_event_items(
                congestion_items=congestion_items,
                traffic_control_items=traffic_control_items,
                service_area_items=service_area_items,
                exit_items=exit_items,
            )
            road_summaries.append(
                {
                    "query_road": query_road,
                    "matched_road_names": matched_road_names,
                    "matched_road_count": len(matched_road_names),
                    "result_count": len(payload),
                    "congestion_count": len(congestion_items),
                    "traffic_control_count": len(traffic_control_items),
                    "service_area_count": len(service_area_items),
                    "exit_count": len(exit_items),
                    "event_count": len(event_items),
                    "congestion_items": congestion_items,
                    "traffic_control_items": traffic_control_items,
                    "service_area_items": service_area_items,
                    "exit_items": exit_items,
                    "event_items": event_items,
                }
            )
        return road_summaries

    @staticmethod
    def _build_route_summaries(
        *,
        query_arguments: dict[str, object],
        per_road_results: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        raw_route_summaries = query_arguments.get("route_summaries")
        if not isinstance(raw_route_summaries, list):
            return []

        road_result_map = {
            str(road_result.get("query_road") or "").strip(): road_result
            for road_result in per_road_results
        }
        route_summaries: list[dict[str, object]] = []
        for route_summary in raw_route_summaries:
            if not isinstance(route_summary, dict):
                continue

            road_details: list[dict[str, object]] = []
            route_congestion_items: list[dict[str, object]] = []
            route_control_items: list[dict[str, object]] = []
            route_service_area_items: list[dict[str, object]] = []
            route_exit_items: list[dict[str, object]] = []

            for road_name in route_summary.get("road_names", []):
                normalized_road_name = str(road_name).strip()
                if not normalized_road_name:
                    continue
                road_result = road_result_map.get(normalized_road_name, {})
                api_result = road_result.get("api_result")
                payload = (
                    [item for item in api_result if isinstance(item, dict)]
                    if isinstance(api_result, list)
                    else []
                )
                congestion_items = TrafficNode._extract_congestion_items(payload)
                traffic_control_items = TrafficNode._extract_traffic_control_items(payload)
                service_area_items = TrafficNode._extract_service_area_items(payload)
                exit_items = TrafficNode._extract_exit_items(payload)
                road_event_items = TrafficNode._build_event_items(
                    congestion_items=congestion_items,
                    traffic_control_items=traffic_control_items,
                    service_area_items=service_area_items,
                    exit_items=exit_items,
                )
                matched_road_names = TrafficNode._deduplicate_strings(
                    str(item.get("roadName") or "").strip() for item in payload
                )
                road_details.append(
                    {
                        "road_name": normalized_road_name,
                        "matched_road_names": matched_road_names,
                        "result_count": len(payload),
                        "congestion_count": len(congestion_items),
                        "traffic_control_count": len(traffic_control_items),
                        "service_area_count": len(service_area_items),
                        "exit_count": len(exit_items),
                        "congestion_items": congestion_items,
                        "traffic_control_items": traffic_control_items,
                        "service_area_items": service_area_items,
                        "exit_items": exit_items,
                        "event_count": len(road_event_items),
                        "event_items": road_event_items,
                    }
                )
                route_congestion_items.extend(congestion_items)
                route_control_items.extend(traffic_control_items)
                route_service_area_items.extend(service_area_items)
                route_exit_items.extend(exit_items)

            route_event_items = TrafficNode._build_event_items(
                congestion_items=route_congestion_items,
                traffic_control_items=route_control_items,
                service_area_items=route_service_area_items,
                exit_items=route_exit_items,
            )

            route_summaries.append(
                {
                    "route_index": route_summary.get("route_index"),
                    "distance": route_summary.get("distance"),
                    "duration": route_summary.get("duration"),
                    "toll": route_summary.get("toll"),
                    "section_count": route_summary.get("section_count"),
                    "road_names": list(route_summary.get("road_names", [])),
                    "road_details": road_details,
                    "has_congestion": bool(route_congestion_items),
                    "has_control": bool(route_control_items),
                    "congestion_count": len(route_congestion_items),
                    "traffic_control_count": len(route_control_items),
                    "service_area_count": len(route_service_area_items),
                    "exit_count": len(route_exit_items),
                    "congestion_items": route_congestion_items,
                    "traffic_control_items": route_control_items,
                    "service_area_items": route_service_area_items,
                    "exit_items": route_exit_items,
                    "event_count": len(route_event_items),
                    "event_items": route_event_items,
                }
            )
        return route_summaries

    @staticmethod
    def _build_event_items(
        *,
        congestion_items: list[dict[str, object]],
        traffic_control_items: list[dict[str, object]],
        service_area_items: list[dict[str, object]],
        exit_items: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        event_items: list[dict[str, object]] = []
        for item in congestion_items:
            direction_label = TrafficNode._resolve_direction_label(item.get("direction_type"))
            event_items.append(
                {
                    "event_category": "congestion",
                    "event_label": "拥堵",
                    "road_name": item.get("road_name"),
                    "road_code": item.get("road_code"),
                    "title": item.get("description") or item.get("status") or "拥堵事件",
                    "description": item.get("description"),
                    "status": item.get("status"),
                    "direction_type": item.get("direction_type"),
                    "direction_label": direction_label,
                    "location_description": item.get("location_description"),
                    "start_time": item.get("start_time"),
                    "end_time": item.get("end_time"),
                    "control_measures": item.get("control_measures"),
                    "event_type": item.get("event_type"),
                    "sub_event_type": item.get("sub_event_type"),
                }
            )
        for item in traffic_control_items:
            event_label = TrafficNode._resolve_control_event_label(item.get("event_type"))
            event_items.append(
                {
                    "event_category": "traffic_control",
                    "event_label": event_label,
                    "road_name": item.get("road_name"),
                    "road_code": item.get("road_code"),
                    "title": item.get("description") or item.get("control_name") or event_label,
                    "description": item.get("description"),
                    "control_name": item.get("control_name"),
                    "control_type": item.get("control_type"),
                    "direction_type": item.get("direction_type"),
                    "direction_label": TrafficNode._resolve_direction_label(item.get("direction_type")),
                    "location_description": item.get("location_description"),
                    "start_time": item.get("start_time"),
                    "end_time": item.get("end_time"),
                    "control_measures": item.get("control_measures"),
                    "event_type": item.get("event_type"),
                    "sub_event_type": item.get("sub_event_type"),
                }
            )
        for item in service_area_items:
            event_items.append(
                {
                    "event_category": "service_area",
                    "event_label": "服务区",
                    "road_name": item.get("road_name"),
                    "road_code": item.get("road_code"),
                    "title": item.get("service_name") or "服务区信息",
                    "description": item.get("description"),
                    "direction_type": item.get("direction_type"),
                    "status_tag": item.get("status_tag"),
                    "direction_label": TrafficNode._resolve_direction_label(item.get("direction_type")),
                    "status_label": TrafficNode._resolve_service_status_label(item.get("status_tag")),
                }
            )
        for item in exit_items:
            event_items.append(
                {
                    "event_category": "toll_station",
                    "event_label": "收费站",
                    "road_name": item.get("road_name"),
                    "road_code": item.get("road_code"),
                    "title": item.get("toll_name") or item.get("exit_name") or "收费站信息",
                    "description": item.get("description"),
                    "entrance_status": item.get("entrance_status"),
                    "entrance_status_label": item.get("entrance_status_label"),
                    "export_status": item.get("export_status"),
                    "export_status_label": item.get("export_status_label"),
                }
            )
        return event_items

    @staticmethod
    def _extract_congestion_items(
        response_payload: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        congestion_items: list[dict[str, object]] = []
        for road in response_payload:
            road_name = str(road.get("roadName") or "").strip()
            road_code = str(road.get("roadGbCode") or "").strip() or None
            congestion_list = road.get("congestionInfoList", [])
            if not isinstance(congestion_list, list):
                continue
            for item in congestion_list:
                if not isinstance(item, dict):
                    continue
                begin_milestone = item.get("beginMilestone")
                end_milestone = item.get("endMilestone")
                congestion_items.append(
                    {
                        "road_name": road_name,
                        "road_code": road_code,
                        "congestion_id": item.get("id"),
                        "description": item.get("description") or item.get("content") or item.get("des"),
                        "status": item.get("status"),
                        "start_time": item.get("startTime") or item.get("beginTime"),
                        "end_time": item.get("endTime") or item.get("expectedEndTime"),
                        "begin_milestone": begin_milestone,
                        "end_milestone": end_milestone,
                        "direction_type": item.get("directionType"),
                        "direction_label": TrafficNode._resolve_direction_label(item.get("directionType")),
                        "event_type": item.get("eventType"),
                        "sub_event_type": item.get("subEventType"),
                        "control_measures": item.get("controlMeasures"),
                        "road_amble_mile": item.get("roadAmbleMile"),
                        "road_id": item.get("roadId"),
                        "location_description": TrafficNode._build_location_description(
                            begin_milestone=begin_milestone,
                            end_milestone=end_milestone,
                            direction_type=item.get("directionType"),
                        ),
                    }
                )
        return congestion_items

    @staticmethod
    def _extract_traffic_control_items(
        response_payload: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        traffic_control_items: list[dict[str, object]] = []
        for road in response_payload:
            road_name = str(road.get("roadName") or "").strip()
            road_code = str(road.get("roadGbCode") or "").strip() or None
            control_list = road.get("trafficControlList", [])
            if not isinstance(control_list, list):
                continue
            for item in control_list:
                if not isinstance(item, dict):
                    continue
                begin_milestone = item.get("beginMilestone")
                end_milestone = item.get("endMilestone")
                traffic_control_items.append(
                    {
                        "road_name": road_name,
                        "road_code": road_code,
                        "control_id": item.get("id"),
                        "control_name": item.get("name") or item.get("controlName"),
                        "control_type": item.get("type") or item.get("controlType"),
                        "description": item.get("description") or item.get("content") or item.get("des"),
                        "start_time": item.get("startTime") or item.get("beginTime"),
                        "end_time": item.get("endTime") or item.get("expectedEndTime"),
                        "begin_milestone": begin_milestone,
                        "end_milestone": end_milestone,
                        "direction_type": item.get("directionType"),
                        "direction_label": TrafficNode._resolve_direction_label(item.get("directionType")),
                        "event_type": item.get("eventType"),
                        "sub_event_type": item.get("subEventType"),
                        "control_measures": item.get("controlMeasures"),
                        "road_amble_mile": item.get("roadAmbleMile"),
                        "road_id": item.get("roadId"),
                        "location_description": TrafficNode._build_location_description(
                            begin_milestone=begin_milestone,
                            end_milestone=end_milestone,
                            direction_type=item.get("directionType"),
                        ),
                    }
                )
        return traffic_control_items

    @staticmethod
    def _extract_service_area_items(
        response_payload: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        service_area_items: list[dict[str, object]] = []
        for road in response_payload:
            road_name = str(road.get("roadName") or "").strip()
            road_code = str(road.get("roadGbCode") or "").strip() or None
            service_area_list = road.get("serviceAreaList", [])
            if not isinstance(service_area_list, list):
                continue
            for item in service_area_list:
                if not isinstance(item, dict):
                    continue
                service_area_items.append(
                    {
                        "road_name": road_name,
                        "road_code": road_code,
                        "service_name": item.get("serviceName"),
                        "service_id": item.get("serviceId"),
                        "direction_type": item.get("directionType"),
                        "direction_label": TrafficNode._resolve_direction_label(item.get("directionType")),
                        "status_tag": item.get("statusTag"),
                        "status_label": TrafficNode._resolve_service_status_label(item.get("statusTag")),
                        "description": item.get("description") or item.get("content"),
                    }
                )
        return service_area_items

    @staticmethod
    def _extract_exit_items(response_payload: list[dict[str, object]]) -> list[dict[str, object]]:
        exit_items: list[dict[str, object]] = []
        for road in response_payload:
            road_name = str(road.get("roadName") or "").strip()
            road_code = str(road.get("roadGbCode") or "").strip() or None
            exit_info_list = road.get("exitInfoList", [])
            if not isinstance(exit_info_list, list):
                continue
            for item in exit_info_list:
                if not isinstance(item, dict):
                    continue
                exit_items.append(
                    {
                        "road_name": road_name,
                        "road_code": road_code,
                        "toll_name": item.get("tollName"),
                        "toll_id": item.get("tollId"),
                        "exit_name": item.get("exitName"),
                        "entrance_status": item.get("entranceStatus"),
                        "entrance_status_label": TrafficNode._resolve_station_status_label(item.get("entranceStatus")),
                        "export_status": item.get("exportStatus"),
                        "export_status_label": TrafficNode._resolve_station_status_label(item.get("exportStatus")),
                        "description": item.get("description") or item.get("content"),
                    }
                )
        return exit_items

    @staticmethod
    def _build_location_description(
        *,
        begin_milestone: object,
        end_milestone: object,
        direction_type: object,
    ) -> str | None:
        parts: list[str] = []
        direction_label = TrafficNode._resolve_direction_label(direction_type)
        if direction_label:
            parts.append(f"方向:{direction_label}")
        elif direction_type not in (None, ""):
            parts.append(f"方向:{direction_type}")
        if begin_milestone not in (None, "") or end_milestone not in (None, ""):
            parts.append(f"桩号:{begin_milestone}-{end_milestone}")
        return " ".join(parts) or None

    @staticmethod
    def _resolve_control_event_label(event_type: object) -> str:
        normalized_event_type = str(event_type or "").strip().lower()
        if normalized_event_type == "construction":
            return "施工"
        if normalized_event_type in {"control", "traffic_control"}:
            return "管制"
        return "交通事件"

    @staticmethod
    def _resolve_direction_label(direction_type: object) -> str | None:
        normalized_direction = str(direction_type or "").strip()
        if not normalized_direction:
            return None
        if normalized_direction in {"00", "0", "双向"}:
            return "双向"
        stripped_direction = normalized_direction.lstrip("0")
        if stripped_direction == "1":
            return "上行"
        if stripped_direction == "2":
            return "下行"
        if normalized_direction in {"上行", "下行"}:
            return normalized_direction
        return normalized_direction

    @staticmethod
    def _resolve_station_status_label(status_code: object) -> str | None:
        normalized_status = str(status_code or "").strip()
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
    def _resolve_service_status_label(status_tag: object) -> str | None:
        normalized_status_tag = str(status_tag or "").strip()
        return normalized_status_tag or None

    @staticmethod
    def _deduplicate_strings(values: Iterable[object]) -> list[str]:
        seen: set[str] = set()
        ordered_values: list[str] = []
        for raw_value in values:
            value = str(raw_value).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered_values.append(value)
        return ordered_values
