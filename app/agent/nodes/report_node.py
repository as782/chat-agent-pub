"""Network report node.

This node normalizes the live-agent `topN` response into a prompt-friendly
`report_context` string and a compact executor result payload.
"""

from __future__ import annotations

from collections.abc import Callable
from json import loads

from app.agent.event_catalog import resolve_event_type_name, resolve_sub_event_type
from app.agent.event_filter import should_filter_report_event
from app.agent.prompts import REPORT_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
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


class ReportNode:
    """LangGraph network report node."""

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or ToolRegistry()

    async def run(self, state: AgentState) -> dict[str, object]:
        """Execute the network overview tool and build report context."""

        step_id = resolve_active_execution_step_id(
            state,
            executor="report",
            default_step_id="report_1",
        )
        resolved_arguments = resolve_step_arguments(state, step_id=step_id, executor="report")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"report_context": None}

        query_arguments = self._build_tool_arguments(resolved_arguments)
        try:
            tool_output = await self._tool_registry.execute_named_tool(
                tool_name="live_network_overview_query",
                arguments=query_arguments,
            )
            parsed_payload = self._parse_tool_output(tool_output)
            response_payload = self._unwrap_network_overview_payload(parsed_payload)
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="report",
                is_success=True,
                raw_result={
                    "query_arguments": dict(query_arguments),
                    "api_result": parsed_payload,
                },
                normalized_result=self._build_normalized_result(
                    response_payload=response_payload,
                ),
                summary="整体路网查询成功。",
            )
            return {
                "report_context": self._build_report_context(
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
        """The live network overview endpoint currently takes no arguments."""

        del resolved_arguments
        return {}

    @staticmethod
    def _parse_tool_output(tool_output: str) -> dict[str, object] | list[dict[str, object]]:
        """Parse the tool JSON output."""

        response_payload = loads(tool_output)
        if isinstance(response_payload, dict):
            return response_payload
        if isinstance(response_payload, list):
            return [item for item in response_payload if isinstance(item, dict)]
        return {}

    @staticmethod
    def _unwrap_network_overview_payload(
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> dict[str, object] | list[dict[str, object]]:
        """Unwrap the standard `{code, data, message}` envelope when needed."""

        if not isinstance(response_payload, dict):
            return response_payload

        data = response_payload.get("data")
        if isinstance(data, dict):
            topn_keys = {"queryTime", "congestionTopN", "majorTopN", "controlTopN", "exitTopN"}
            if any(key in data for key in topn_keys):
                return data
        return response_payload

    @staticmethod
    def _build_normalized_result(
        *,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> dict[str, object]:
        """Extract the key report fields for downstream consumers."""

        if isinstance(response_payload, list):
            return {
                "record_count": len(response_payload),
                "query_time": None,
                "congestion_total_mile": None,
                "congestion_top_count": 0,
                "major_top_count": 0,
                "accident_top_count": 0,
                "control_top_count": 0,
                "exit_top_count": 0,
                "congestion_top_items": [],
                "major_top_items": [],
                "accident_top_items": [],
                "control_top_items": [],
                "exit_top_items": [],
            }

        query_time = ReportNode._string_or_placeholder(response_payload.get("queryTime"))
        congestion_total_mile = ReportNode._extract_congestion_total_mile(response_payload)
        congestion_top_items = ReportNode._extract_payload_items(response_payload, "congestionTopN")
        major_top_items = ReportNode._extract_payload_items(response_payload, "majorTopN")
        accident_top_items = ReportNode._extract_payload_items(response_payload, "accidentTopN")
        control_top_items = ReportNode._extract_payload_items(response_payload, "controlTopN")
        exit_top_items = ReportNode._extract_payload_items(response_payload, "exitTopN")

        # 对拥堵事件进行筛选，过滤掉不需要的事件类型
        filtered_congestion_items = [
            item for item in congestion_top_items 
            if not ReportNode._should_filter_event(item)
        ]
        filtered_major_items = [
            item for item in major_top_items
            if not ReportNode._should_filter_major_event(item)
        ]
        
        # 对主线管制事件进行筛选
        filtered_control_items = [
            item for item in control_top_items 
            if not ReportNode._should_filter_event(item)
        ]
        
        # 对事故事件进行筛选
        filtered_accident_items = [
            item for item in accident_top_items 
            if not ReportNode._should_filter_event(item)
        ]
        
        # 收费站管制通常不需要过滤，保持原样
        filtered_exit_items = exit_top_items

        return {
            "record_count": 1,
            "query_time": query_time,
            "congestion_total_mile": congestion_total_mile,
            "congestion_top_count": len(filtered_congestion_items),
            "major_top_count": len(filtered_major_items),
            "accident_top_count": len(filtered_accident_items),
            "control_top_count": len(filtered_control_items),
            "exit_top_count": len(filtered_exit_items),
            "congestion_top_items": filtered_congestion_items,
            "major_top_items": filtered_major_items,
            "accident_top_items": filtered_accident_items,
            "control_top_items": filtered_control_items,
            "exit_top_items": filtered_exit_items,
        }

    @staticmethod
    def _string_or_placeholder(value: object | None) -> str:
        """Normalize any value to a readable string."""

        if value is None:
            return "未知"
        text = str(value).strip()
        return text if text else "未知"

    @staticmethod
    def _format_number(value: object | None) -> str:
        """Normalize a numeric value for display."""

        if value is None:
            return "未知"
        if isinstance(value, str):
            text = value.strip()
            return text if text else "未知"
        return str(value)

    @classmethod
    def _format_direction(cls, *candidates: object | None) -> str:
        """Normalize direction codes and labels into Chinese text."""

        mapping = {
            "0": "双向",
            "00": "双向",
            "双向": "双向",
            "1": "上行",
            "01": "上行",
            "100700": "上行",
            "上行": "上行",
            "2": "下行",
            "02": "下行",
            "100701": "下行",
            "下行": "下行",
        }
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if not text:
                continue
            if text in mapping:
                return mapping[text]
            if any("\u4e00" <= char <= "\u9fff" for char in text):
                return text
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                return text
        return "未知"

    @staticmethod
    def _format_milestone(value: object | None) -> str:
        """Normalize milestone values."""

        if value is None:
            return "未知"
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("K"):
                text = text[1:]
            return text if text else "未知"
        return str(value)

    @staticmethod
    def _format_entrance_label(value: object | None) -> str:
        """Normalize entrance/exit codes."""

        mapping = {
            "0": "出口",
            "1": "入口",
            "出口": "出口",
            "入口": "入口",
        }
        if value is None:
            return "未知"
        text = str(value).strip()
        if not text:
            return "未知"
        return mapping.get(text, text)

    @classmethod
    def _format_event_category(
        cls,
        event_class: object | None,
        event_type: object | None,
        sub_event_type_id: object | None = None,
    ) -> str:
        """Format the event classification using the provided code mapping."""

        class_label = cls._string_or_placeholder(event_class)
        type_label = cls._string_or_placeholder(event_type)

        resolved_class_label = resolve_event_type_name(class_label) or class_label
        resolved_type_label = resolve_event_type_name(type_label) or type_label
        if resolved_type_label == "未知" or resolved_type_label == resolved_class_label:
            category = resolved_class_label
        else:
            category = f"{resolved_class_label}（{resolved_type_label}）"

        sub_event = resolve_sub_event_type(sub_event_type_id)
        if sub_event is None or sub_event["name"] == "无":
            return category
        return f"{category} / 小类 {sub_event['name']}"

    @classmethod
    def _extract_payload_items(
        cls,
        response_payload: dict[str, object] | list[dict[str, object]],
        key: str,
    ) -> list[dict[str, object]]:
        """Extract a topN list from the normalized payload."""

        if not isinstance(response_payload, dict):
            return []
        items = response_payload.get(key, [])
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _extract_congestion_total_mile(response_payload: dict[str, object]) -> object | None:
        congestion_payload = response_payload.get("congestion")
        if isinstance(congestion_payload, dict):
            total_mile = congestion_payload.get("totalMile")
            if total_mile is not None:
                return total_mile
        if "congestionTotalMile" in response_payload:
            return response_payload.get("congestionTotalMile")
        return None

    @classmethod
    def _build_identity(cls, road_code: object | None, road_name: object | None) -> str:
        """Combine road code and road name into one label."""

        code_text = cls._string_or_placeholder(road_code)
        name_text = cls._string_or_placeholder(road_name)
        if code_text == "未知" and name_text == "未知":
            return "未知"
        if code_text == "未知":
            return name_text
        if name_text == "未知":
            return code_text
        return f"{code_text} / {name_text}"

    @classmethod
    def _build_item_section(
        cls,
        title: str,
        items: list[dict[str, object]],
        formatter: Callable[[dict[str, object]], str],
    ) -> str:
        """Render one topN section."""

        lines = [f"{title}（{len(items)}条）："]
        if not items:
            lines.append("- 暂无")
            return "\n".join(lines)
        for item in items:
            lines.append(formatter(item))
        return "\n".join(lines)

    @classmethod
    def _build_congestion_line(cls, item: dict[str, object]) -> str:
        """Render one congestion topN record."""

        road_identity = cls._build_identity(
            item.get("roadGBCode") or item.get("roadGbCode") or item.get("roadCode"),
            item.get("roadName"),
        )
        direction = cls._format_direction(
            item.get("directionName"),
            item.get("directionType"),
            item.get("direction"),
        )
        location = (
            f"K{cls._format_milestone(item.get('beginMilestoneStr') or item.get('beginMilestone'))}"
            f"-K{cls._format_milestone(item.get('endMilestoneStr') or item.get('endMilestone'))}"
        )
        expected_end_time = cls._string_or_placeholder(
            item.get("expectedTime")
            or item.get("expectedEndTime")
            or item.get("endTime")
        )
        event_category = cls._format_event_category(
            item.get("eventClass"),
            item.get("eventType"),
            item.get("subEventTypeId"),
        )
        parts = [
            f"- {road_identity}",
            f"方向 {direction}",
            f"区间 {location}",
            f"缓行里程 {cls._format_number(item.get('roadAmbleMile'))} 公里",
            f"开始 {cls._string_or_placeholder(item.get('beginTime'))}",
            f"预计结束 {expected_end_time}",
            f"事件分类 {event_category}",
            f"管制措施 {cls._string_or_placeholder(item.get('controlMeasures'))}",
            f"现场情况 {cls._string_or_placeholder(item.get('situationRemark'))}",
            f"占道情况 {cls._string_or_placeholder(item.get('jeeves'))}",
            f"描述 {cls._string_or_placeholder(item.get('des'))}",
        ]
        return " | ".join(parts)

    @classmethod
    def _build_control_line(cls, item: dict[str, object]) -> str:
        """Render one mainline control topN record."""

        road_identity = cls._build_identity(
            item.get("roadGBCode") or item.get("roadGbCode") or item.get("roadCode"),
            item.get("roadName"),
        )
        direction = cls._format_direction(
            item.get("directionName"),
            item.get("directionType"),
            item.get("direction"),
        )
        station_name = cls._string_or_placeholder(
            item.get("tollName") or item.get("entranceName") or item.get("controlName")
        )
        entrance_label = cls._format_entrance_label(item.get("entrance"))
        control_type = cls._string_or_placeholder(
            item.get("controlTypeName") or item.get("controlType")
        )
        limit_measure = cls._string_or_placeholder(
            item.get("limitMeasureTypeName") or item.get("controlMeasures")
        )
        end_time = cls._string_or_placeholder(
            item.get("endTime") or item.get("expectedTime") or item.get("expectedEndTime")
        )
        description = cls._string_or_placeholder(item.get("des") or item.get("description"))
        parts = [
            f"- {road_identity}",
            f"方向 {direction}",
            f"收费站 {station_name}",
            f"出入口 {entrance_label}",
            f"管制类型 {control_type}",
            f"措施 {limit_measure}",
            f"开始 {cls._string_or_placeholder(item.get('startTime') or item.get('beginTime'))}",
            f"结束 {end_time}",
            f"描述 {description}",
        ]
        return " | ".join(parts)

    @classmethod
    def _build_exit_line(cls, item: dict[str, object]) -> str:
        """Render one toll-station control topN record."""

        road_identity = cls._build_identity(
            item.get("roadGBCode") or item.get("roadGbCode") or item.get("roadCode"),
            item.get("roadName"),
        )
        direction = cls._format_direction(
            item.get("directionName"),
            item.get("directionType"),
            item.get("direction"),
        )
        station_name = cls._string_or_placeholder(
            item.get("tollName") or item.get("exitName") or item.get("entranceName")
        )
        entrance_label = cls._format_entrance_label(item.get("entrance"))
        control_type = cls._string_or_placeholder(
            item.get("controlTypeName") or item.get("controlType")
        )
        limit_measure = cls._string_or_placeholder(
            item.get("limitMeasureTypeName") or item.get("controlMeasures")
        )
        end_time = cls._string_or_placeholder(
            item.get("endTime") or item.get("expectedTime") or item.get("expectedEndTime")
        )
        description = cls._string_or_placeholder(item.get("des") or item.get("description"))
        parts = [
            f"- {road_identity}",
            f"方向 {direction}",
            f"收费站 {station_name}",
            f"出入口 {entrance_label}",
            f"管制状态 {control_type}",
            f"措施 {limit_measure}",
            f"开始 {cls._string_or_placeholder(item.get('startTime') or item.get('beginTime'))}",
            f"结束 {end_time}",
            f"描述 {description}",
        ]
        return " | ".join(parts)

    @staticmethod
    def _should_filter_event(item: dict[str, object]) -> bool:
        """判断是否应该过滤掉该事件，基于事件分类码值。
        
        根据需求，需要过滤掉交通事故(04)和车辆故障(07)等事件，
        以及10110硬路肩开放的管制类型。
        """
        return should_filter_report_event(item)

    @staticmethod
    def _should_filter_major_event(item: dict[str, object]) -> bool:
        """Filter majorTopN without treating eventClass=07 as vehicle fault."""

        event_type = ReportNode._string_or_placeholder(item.get("eventType"))
        control_type = ReportNode._string_or_placeholder(item.get("controlType"))
        return event_type in {"01", "97"} or control_type == "10110"

    @classmethod
    def _build_compact_report_context(
        cls,
        *,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> str:
        """Build the report content used by the answer prompt."""

        if not isinstance(response_payload, dict):
            lines = [
                "查询时间：未知",
                "拥堵汇总（0条）：",
                "- 暂无",
                "",
                "重大事件（0条）：",
                "- 暂无",
                "",
                "主线管制（0条）：",
                "- 暂无",
                "",
                "收费站管制（0条）：",
                "- 暂无",
            ]
            return "\n".join(lines)

        query_time = cls._string_or_placeholder(response_payload.get("queryTime"))
        congestion_total_mile = cls._extract_congestion_total_mile(response_payload)
        congestion_items = cls._extract_payload_items(response_payload, "congestionTopN")
        major_items = cls._extract_payload_items(response_payload, "majorTopN")
        control_items = cls._extract_payload_items(response_payload, "controlTopN")
        exit_items = cls._extract_payload_items(response_payload, "exitTopN")
        
        # 对拥堵事件进行筛选，过滤掉不需要的事件类型
        filtered_congestion_items = [
            item for item in congestion_items 
            if not cls._should_filter_event(item)
        ]
        filtered_major_items = [
            item for item in major_items
            if not cls._should_filter_major_event(item)
        ]
        
        # 对主线管制事件进行筛选
        filtered_control_items = [
            item for item in control_items 
            if not cls._should_filter_event(item)
        ]
        
        # 收费站管制通常不需要过滤，保持原样
        filtered_exit_items = exit_items

        congestion_section = cls._build_item_section(
            "拥堵汇总",
            filtered_congestion_items,
            cls._build_congestion_line,
        )
        major_section = cls._build_item_section(
            "重大事件",
            filtered_major_items,
            cls._build_congestion_line,
        )
        control_section = cls._build_item_section(
            "主线管制",
            filtered_control_items,
            cls._build_control_line,
        )
        exit_section = cls._build_item_section(
            "收费站管制",
            filtered_exit_items,
            cls._build_exit_line,
        )

        lines = [f"查询时间：{query_time}"]
        if congestion_total_mile is not None:
            lines.append(f"拥堵总里程：{cls._format_number(congestion_total_mile)} 公里")
        lines.append(congestion_section)
        lines.append("")
        lines.append(major_section)
        lines.append("")
        lines.append(control_section)
        lines.append("")
        lines.append(exit_section)
        return "\n".join(lines)

    @staticmethod
    def _build_report_context(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> str:
        """Build the final system prompt snippet for network reports."""

        del resolved_arguments
        return "\n".join(
            [
                REPORT_CONTEXT_PROMPT_PREFIX,
                ReportNode._build_compact_report_context(response_payload=response_payload),
            ]
        )
