"""Service area business node."""

from __future__ import annotations

from collections.abc import Iterable
from json import loads

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
    """LangGraph service area node."""

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or ToolRegistry()

    async def run(self, state: AgentState) -> dict[str, object]:
        """Execute the service query tool and build service context."""

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
                "service_context": self._build_service_context(response_payload=response_payload),
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
        """Convert structured arguments into service tool arguments."""

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
        """Prefer structured service fields before falling back to raw query text."""

        arguments = resolved_arguments.arguments
        raw_query_terms = arguments.get("service_query_terms")
        if isinstance(raw_query_terms, list):
            for raw_term in raw_query_terms:
                value = str(raw_term).strip()
                if value:
                    return value
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
        """Parse the JSON output from the service tool."""

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
        """Extract the useful service fields from the API payload."""

        first_result = response_payload[0] if response_payload else {}
        charge_items = ServiceNode._extract_charge_items(first_result)
        commercial_items = ServiceNode._extract_commercial_items(first_result)
        tags = ServiceNode._extract_tags(first_result)
        return {
            "keyword": query_arguments.get("keyword"),
            "result_count": len(response_payload),
            "service_name": first_result.get("serviceName"),
            "road_name": first_result.get("roadName"),
            "direction": first_result.get("direction") or first_result.get("directionType"),
            "direction_name": first_result.get("directionName"),
            "milestone": first_result.get("milestone"),
            "status_tag": first_result.get("statusTag"),
            "has_charging": bool(charge_items),
            "charge_summary_count": len(charge_items),
            "charge_brand_count": len(charge_items),
            "charge_items": charge_items,
            "commercial_count": len(commercial_items),
            "commercial_items": commercial_items,
            "tag_count": len(tags),
            "tags": tags,
        }

    @staticmethod
    def _build_service_context(*, response_payload: list[dict[str, object]]) -> str:
        """Build a compact text context instead of dumping raw JSON."""

        service_blocks = ServiceNode._build_service_context_blocks(response_payload)
        if not service_blocks:
            return "\n".join(
                [
                    SERVICE_CONTEXT_PROMPT_PREFIX.rstrip(),
                    "暂无可用服务区信息。",
                ]
            )

        return "\n\n".join([SERVICE_CONTEXT_PROMPT_PREFIX.rstrip(), *service_blocks])

    @staticmethod
    def _build_service_context_blocks(response_payload: list[dict[str, object]]) -> list[str]:
        service_items = [item for item in response_payload if isinstance(item, dict)]
        if not service_items:
            return []

        blocks = [f"服务区摘要：共 {len(service_items)} 条"]
        for index, service in enumerate(service_items[:3], start=1):
            blocks.append(ServiceNode._build_service_block(service, index=index))

        if len(service_items) > 3:
            blocks.append(f"其余 {len(service_items) - 3} 条服务区信息已省略")
        return blocks

    @classmethod
    def _build_service_block(cls, service: dict[str, object], *, index: int) -> str:
        service_name = cls._string_or_placeholder(service.get("serviceName"), "未知服务区")
        road_name = cls._string_or_placeholder(service.get("roadName"), "未知高速")
        road_code = cls._string_or_placeholder(service.get("roadGbCode"), "")
        road_display = road_name
        if road_code and road_code not in road_display:
            road_display = f"{road_display}（{road_code}）"

        direction_text = cls._format_service_direction(service)
        milestone = cls._string_or_placeholder(service.get("milestone"), "未知")
        status_tag = cls._string_or_placeholder(service.get("statusTag"), "未知")
        charge_lines = cls._build_charge_summary_lines(service)
        tag_line = cls._build_tag_line(service.get("tags"))
        commercial_lines = cls._build_commercial_summary_lines(service)
        remark_lines = cls._build_service_remark_lines(service)

        lines = [
            f"服务区 {index}：",
            f"- 基本信息：{service_name}（{road_display}，{direction_text}，{milestone}，{status_tag}）",
        ]

        if charge_lines:
            lines.append("充电情况：")
            lines.extend(f"  - {line}" for line in charge_lines)

        if tag_line:
            lines.append(f"其他配套：{tag_line}")

        if commercial_lines:
            lines.append("商业配套：")
            lines.extend(f"  - {line}" for line in commercial_lines)

        if remark_lines:
            lines.append("补充提醒：")
            lines.extend(f"  - {line}" for line in remark_lines)

        return "\n".join(lines)

    @classmethod
    def _extract_charge_items(cls, first_result: dict[str, object]) -> list[dict[str, object]]:
        charge_list = first_result.get("chargeList", [])
        if not isinstance(charge_list, list):
            return []

        charge_items: list[dict[str, object]] = []
        for index, item in enumerate(charge_list, start=1):
            if not isinstance(item, dict):
                continue
            charge_items.append(
                {
                    "index": index,
                    "brand": item.get("manufacturerName") or item.get("brand"),
                    "pile_count": item.get("pileCount"),
                    "power": item.get("power"),
                    "status": item.get("status"),
                    "total_charging_num": item.get("totalChargingNum"),
                    "total_dc_charging_num": item.get("totalDCChargingNum"),
                    "total_ac_charging_num": item.get("totalACChargingNum"),
                    "total_free_charging_num": item.get("totalFreeChargingNum"),
                    "total_free_dc_charging_num": item.get("totalFreeDCChargingNum"),
                    "total_free_ac_charging_num": item.get("totalFreeACChargingNum"),
                }
            )
        return charge_items

    @classmethod
    def _build_charge_summary_lines(cls, service: dict[str, object]) -> list[str]:
        charge_items = cls._extract_charge_items(service)
        if not charge_items:
            return []

        lines: list[str] = []
        for charge_item in charge_items[:2]:
            prefix = ""
            if len(charge_items) > 1:
                prefix = f"第{charge_item.get('index')}组："

            has_count_fields = any(
                charge_item.get(key) is not None
                for key in (
                    "total_charging_num",
                    "total_dc_charging_num",
                    "total_ac_charging_num",
                    "total_free_charging_num",
                    "total_free_dc_charging_num",
                    "total_free_ac_charging_num",
                )
            )
            if has_count_fields:
                total_charging_num = cls._string_or_placeholder(
                    charge_item.get("total_charging_num"),
                    "未知",
                )
                total_dc_charging_num = cls._string_or_placeholder(
                    charge_item.get("total_dc_charging_num"),
                    "未知",
                )
                total_ac_charging_num = cls._string_or_placeholder(
                    charge_item.get("total_ac_charging_num"),
                    "未知",
                )
                total_free_charging_num = cls._string_or_placeholder(
                    charge_item.get("total_free_charging_num"),
                    "未知",
                )
                total_free_dc_charging_num = cls._string_or_placeholder(
                    charge_item.get("total_free_dc_charging_num"),
                    "未知",
                )
                total_free_ac_charging_num = cls._string_or_placeholder(
                    charge_item.get("total_free_ac_charging_num"),
                    "未知",
                )
                lines.append(
                    f"{prefix}总充电桩：{total_charging_num} 个，快充：{total_dc_charging_num} 个，"
                    f"慢充：{total_ac_charging_num} 个，空闲充电桩：{total_free_charging_num} 个，"
                    f"空闲快充：{total_free_dc_charging_num} 个，空闲慢充：{total_free_ac_charging_num} 个"
                )
            else:
                brand = cls._string_or_placeholder(charge_item.get("brand"), "未知")
                pile_count = cls._string_or_placeholder(charge_item.get("pile_count"), "未知")
                power = cls._string_or_placeholder(charge_item.get("power"), "未知")
                status = cls._string_or_placeholder(charge_item.get("status"), "未知")
                lines.append(
                    f"{prefix}充电品牌：{brand}，桩数：{pile_count}，功率：{power}，状态：{status}"
                )

        if len(charge_items) > 2:
            lines.append(f"其余 {len(charge_items) - 2} 组充电统计已省略")

        lines.append("空闲桩数量仅表示统计口径，不等于实时占用状态。")
        return lines

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
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "business_start_time": item.get("businessStartTime"),
                    "business_end_time": item.get("businessEndTime"),
                }
            )
        return commercial_items

    @classmethod
    def _build_commercial_summary_lines(cls, service: dict[str, object]) -> list[str]:
        commercial_list = service.get("commercialList", [])
        if not isinstance(commercial_list, list):
            return []

        commercial_lines: list[str] = []
        seen_names: set[str] = set()
        for item in commercial_list:
            if not isinstance(item, dict):
                continue
            name = cls._string_or_placeholder(item.get("name"), "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            start_time = cls._string_or_placeholder(item.get("businessStartTime"), "未知")
            end_time = cls._string_or_placeholder(item.get("businessEndTime"), "未知")
            commercial_lines.append(f"{name}（{start_time}-{end_time}）")
            if len(commercial_lines) >= 5:
                break
        return commercial_lines

    @classmethod
    def _build_service_remark_lines(cls, service: dict[str, object]) -> list[str]:
        remark_lines: list[str] = []
        status_tag = cls._string_or_placeholder(service.get("statusTag"), "")
        if status_tag and status_tag != "正常":
            remark_lines.append(f"服务区状态：{status_tag}")
        if not cls._extract_charge_items(service):
            remark_lines.append("未返回充电统计信息。")
        return remark_lines

    @classmethod
    def _build_tag_line(cls, tags: object) -> str | None:
        if not isinstance(tags, list):
            return None
        normalized_tags = cls._deduplicate_strings(tags)
        if not normalized_tags:
            return None
        return "、".join(normalized_tags[:8])

    @classmethod
    def _format_service_direction(cls, service: dict[str, object]) -> str:
        direction_name = cls._string_or_placeholder(service.get("directionName"), "")
        direction = cls._string_or_placeholder(service.get("direction"), "")
        if direction_name and direction and direction_name != direction:
            return f"{direction_name} / {direction}"
        return direction_name or direction or "未知"

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

    @staticmethod
    def _string_or_placeholder(value: object | None, placeholder: str) -> str:
        if value is None:
            return placeholder
        text = str(value).strip()
        return text or placeholder
