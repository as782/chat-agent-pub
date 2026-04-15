"""路况业务节点模块。

负责把当前路况问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问实时路况接口。
"""

from __future__ import annotations

from collections.abc import Iterable
from json import dumps, loads

from app.agent.prompts import TRAFFIC_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
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
from app.tools.registry import ToolRegistry


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

        queried_roads = self._resolve_queried_roads(
            state=state,
            step_id=step_id,
            resolved_arguments=resolved_arguments,
        )
        road = queried_roads[0] if queried_roads else ""
        query_arguments: dict[str, object] = {
            "road": road,
            "queried_roads": queried_roads,
        }
        for key in ("target", "direction", "toll_station", "road_name", "road_code", "query"):
            value = resolved_arguments.arguments.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            query_arguments[key] = value
        return query_arguments

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
                    road_names = dependency_result.normalized_result.get("road_names")
                    if isinstance(road_names, list):
                        normalized_road_names = self._deduplicate_strings(road_names)
                        if normalized_road_names:
                            return normalized_road_names

        resolved_roads = resolved_arguments.arguments.get("roads")
        if isinstance(resolved_roads, list):
            normalized_road_names = self._deduplicate_strings(resolved_roads)
            if normalized_road_names:
                return normalized_road_names

        fallback_road = self._resolve_single_road_argument(resolved_arguments)
        return [fallback_road] if fallback_road else []

    @staticmethod
    def _resolve_single_road_argument(resolved_arguments: ResolvedArguments) -> str:
        """Prefer structured LLM/planner road fields before falling back to raw text."""

        arguments = resolved_arguments.arguments
        for key in ("road_name", "road_code", "road", "target", "query"):
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
        queried_roads = query_arguments.get("queried_roads")
        road_summaries = TrafficNode._build_road_summaries(per_road_results)
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
            "road_summaries": road_summaries,
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
            road_summaries.append(
                {
                    "query_road": query_road,
                    "matched_road_names": matched_road_names,
                    "matched_road_count": len(matched_road_names),
                    "result_count": len(payload),
                    "congestion_count": len(congestion_items),
                    "traffic_control_count": len(traffic_control_items),
                }
            )
        return road_summaries

    @staticmethod
    def _extract_congestion_items(
        response_payload: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        congestion_items: list[dict[str, object]] = []
        for road in response_payload:
            road_name = str(road.get("roadName") or "").strip()
            congestion_list = road.get("congestionInfoList", [])
            if not isinstance(congestion_list, list):
                continue
            for item in congestion_list:
                if not isinstance(item, dict):
                    continue
                congestion_items.append(
                    {
                        "road_name": road_name,
                        "congestion_id": item.get("id"),
                        "description": item.get("description") or item.get("content"),
                        "status": item.get("status"),
                        "start_time": item.get("startTime"),
                        "end_time": item.get("endTime"),
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
            control_list = road.get("trafficControlList", [])
            if not isinstance(control_list, list):
                continue
            for item in control_list:
                if not isinstance(item, dict):
                    continue
                traffic_control_items.append(
                    {
                        "road_name": road_name,
                        "control_id": item.get("id"),
                        "control_name": item.get("name") or item.get("controlName"),
                        "control_type": item.get("type") or item.get("controlType"),
                        "description": item.get("description") or item.get("content"),
                        "start_time": item.get("startTime"),
                        "end_time": item.get("endTime"),
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
            service_area_list = road.get("serviceAreaList", [])
            if not isinstance(service_area_list, list):
                continue
            for item in service_area_list:
                if not isinstance(item, dict):
                    continue
                service_area_items.append(
                    {
                        "road_name": road_name,
                        "service_name": item.get("serviceName"),
                        "status_tag": item.get("statusTag"),
                        "description": item.get("description") or item.get("content"),
                    }
                )
        return service_area_items

    @staticmethod
    def _extract_exit_items(response_payload: list[dict[str, object]]) -> list[dict[str, object]]:
        exit_items: list[dict[str, object]] = []
        for road in response_payload:
            road_name = str(road.get("roadName") or "").strip()
            exit_info_list = road.get("exitInfoList", [])
            if not isinstance(exit_info_list, list):
                continue
            for item in exit_info_list:
                if not isinstance(item, dict):
                    continue
                exit_items.append(
                    {
                        "road_name": road_name,
                        "toll_name": item.get("tollName"),
                        "exit_name": item.get("exitName"),
                        "description": item.get("description") or item.get("content"),
                    }
                )
        return exit_items

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
