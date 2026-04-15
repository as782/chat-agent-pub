"""服务区业务节点模块。

负责在服务区类问题中直接调用标准工具查询服务区、充电和商业配套信息，
并把结果整理为可注入回答节点的上下文。
"""

from __future__ import annotations

from collections.abc import Iterable
from json import dumps, loads

from app.agent.prompts import SERVICE_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
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


class ServiceNode:
    """LangGraph 服务区业务节点。"""

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or ToolRegistry()

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行服务区查询工具并生成服务区业务上下文。"""

        step_id = resolve_active_execution_step_id(
            state,
            executor="service",
            default_step_id="service_1",
        )
        resolved_arguments = resolve_step_arguments(state, step_id=step_id, executor="service")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"service_context": None}
        query_arguments = self._build_tool_arguments(
            state=state,
            step_id=step_id,
            resolved_arguments=resolved_arguments,
        )
        try:
            tool_output = await self._tool_registry.execute_named_tool(
                tool_name="live_service_query",
                arguments={"keyword": str(query_arguments.get("keyword") or "")},
            )
            response_payload = self._parse_tool_output(tool_output)
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="service",
                is_success=True,
                raw_result={
                    "query_arguments": dict(query_arguments),
                    "api_result": response_payload,
                },
                normalized_result=self._build_normalized_result(
                    query_arguments=query_arguments,
                    response_payload=response_payload,
                ),
                summary=f"服务区查询成功，命中 {len(response_payload)} 条结果。",
            )
            return {
                "service_context": self._build_service_context(
                    query_arguments=query_arguments,
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

    def _build_tool_arguments(
        self,
        *,
        state: AgentState,
        step_id: str,
        resolved_arguments: ResolvedArguments,
    ) -> dict[str, object]:
        """把结构化参数转换为服务区查询工具参数。"""

        service_area_names = self._resolve_service_area_names(
            state=state,
            step_id=step_id,
        )
        keyword = (
            service_area_names[0]
            if service_area_names
            else self._resolve_service_keyword(resolved_arguments)
        )
        return {
            "keyword": keyword,
            "service_area_names": service_area_names,
        }

    @staticmethod
    def _resolve_service_keyword(resolved_arguments: ResolvedArguments) -> str:
        """Prefer structured service fields before falling back to the raw query text."""

        arguments = resolved_arguments.arguments
        for key in ("service_name", "keyword", "road_name", "facility_type", "query"):
            value = str(arguments.get(key) or "").strip()
            if value:
                return value
        return ""

    def _resolve_service_area_names(
        self,
        *,
        state: AgentState,
        step_id: str,
    ) -> list[str]:
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
            service_area_names = dependency_result.normalized_result.get("service_area_names")
            if isinstance(service_area_names, list):
                normalized_service_area_names = self._deduplicate_strings(service_area_names)
                if normalized_service_area_names:
                    return normalized_service_area_names
        return []

    @staticmethod
    def _parse_tool_output(tool_output: str) -> list[dict[str, object]]:
        """解析服务区工具返回的 JSON 字符串。"""

        response_payload = loads(tool_output)
        if isinstance(response_payload, list):
            return [item for item in response_payload if isinstance(item, dict)]
        return []

    @staticmethod
    def _build_normalized_result(
        *,
        query_arguments: dict[str, object],
        response_payload: list[dict[str, object]],
    ) -> dict[str, object]:
        """提取服务区查询结果中的完整业务字段。"""

        first_result = response_payload[0] if response_payload else {}
        charge_items = ServiceNode._extract_charge_items(first_result)
        commercial_items = ServiceNode._extract_commercial_items(first_result)
        tags = ServiceNode._extract_tags(first_result)
        return {
            "keyword": query_arguments.get("keyword"),
            "result_count": len(response_payload),
            "service_name": first_result.get("serviceName"),
            "road_name": first_result.get("roadName"),
            "status_tag": first_result.get("statusTag"),
            "has_charging": bool(charge_items),
            "charge_brand_count": len(charge_items),
            "charge_items": charge_items,
            "commercial_count": len(commercial_items),
            "commercial_items": commercial_items,
            "tag_count": len(tags),
            "tags": tags,
        }

    @staticmethod
    def _build_service_context(
        *,
        query_arguments: dict[str, object],
        response_payload: list[dict[str, object]],
    ) -> str:
        """把结构化参数和接口返回转为服务区类 system 上下文。"""

        return "\n".join(
            [
                SERVICE_CONTEXT_PROMPT_PREFIX,
                dumps(
                    {
                        "query_arguments": dict(query_arguments),
                        "api_result": response_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )

    @staticmethod
    def _extract_charge_items(first_result: dict[str, object]) -> list[dict[str, object]]:
        charge_list = first_result.get("chargeList", [])
        if not isinstance(charge_list, list):
            return []
        charge_items: list[dict[str, object]] = []
        for item in charge_list:
            if not isinstance(item, dict):
                continue
            charge_items.append(
                {
                    "brand": item.get("manufacturerName") or item.get("brand"),
                    "pile_count": item.get("pileCount"),
                    "power": item.get("power"),
                    "status": item.get("status"),
                }
            )
        return charge_items

    @staticmethod
    def _extract_commercial_items(first_result: dict[str, object]) -> list[dict[str, object]]:
        commercial_list = first_result.get("commercialList", [])
        if not isinstance(commercial_list, list):
            return []
        commercial_items: list[dict[str, object]] = []
        for item in commercial_list:
            if not isinstance(item, dict):
                continue
            commercial_items.append(
                {
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "status": item.get("status"),
                }
            )
        return commercial_items

    @staticmethod
    def _extract_tags(first_result: dict[str, object]) -> list[str]:
        tags = first_result.get("tags", [])
        if not isinstance(tags, list):
            return []
        return ServiceNode._deduplicate_strings(tags)

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
