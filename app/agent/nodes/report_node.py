"""路网报告业务节点模块。

负责把路网汇总和报表类问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做报表任务规范化，不直接访问外部数据接口。
"""

from __future__ import annotations

from json import loads

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
    """LangGraph 路网报告业务节点。"""

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or ToolRegistry()

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行整体路网查询工具并生成路网报告上下文。"""

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
            response_payload = self._parse_tool_output(tool_output)
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="report",
                is_success=True,
                raw_result={
                    "query_arguments": dict(query_arguments),
                    "api_result": response_payload,
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
        """把结构化参数转换为整体路网查询工具参数。"""

        del resolved_arguments
        return {}

    @staticmethod
    def _parse_tool_output(tool_output: str) -> dict[str, object] | list[dict[str, object]]:
        """解析整体路网工具返回的 JSON 字符串。"""

        response_payload = loads(tool_output)
        if isinstance(response_payload, dict):
            return response_payload
        if isinstance(response_payload, list):
            return [item for item in response_payload if isinstance(item, dict)]
        return {}

    @staticmethod
    def _build_normalized_result(
        *,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> dict[str, object]:
        """提取整体路网查询结果中的完整业务字段。"""

        if isinstance(response_payload, list):
            record_count = len(response_payload)
            congestion_total_mile = None
            query_time = None
            congestion_top_count = 0
            accident_top_count = 0
            control_top_count = 0
            congestion_top_items: list[dict[str, object]] = []
            accident_top_items: list[dict[str, object]] = []
            control_top_items: list[dict[str, object]] = []
        else:
            congestion_payload = response_payload.get("congestion", {})
            congestion_total_mile = (
                congestion_payload.get("totalMile")
                if isinstance(congestion_payload, dict)
                else None
            )
            query_time = response_payload.get("queryTime")
            congestion_top = response_payload.get("congestionTopN", [])
            accident_top = response_payload.get("accidentTopN", [])
            control_top = response_payload.get("controlTopN", [])
            congestion_top_count = len(congestion_top) if isinstance(congestion_top, list) else 0
            accident_top_count = len(accident_top) if isinstance(accident_top, list) else 0
            control_top_count = len(control_top) if isinstance(control_top, list) else 0
            congestion_top_items = (
                [item for item in congestion_top if isinstance(item, dict)]
                if isinstance(congestion_top, list)
                else []
            )
            accident_top_items = (
                [item for item in accident_top if isinstance(item, dict)]
                if isinstance(accident_top, list)
                else []
            )
            control_top_items = (
                [item for item in control_top if isinstance(item, dict)]
                if isinstance(control_top, list)
                else []
            )
            record_count = 1
        return {
            "record_count": record_count,
            "query_time": query_time,
            "congestion_total_mile": congestion_total_mile,
            "congestion_top_count": congestion_top_count,
            "accident_top_count": accident_top_count,
            "control_top_count": control_top_count,
            "congestion_top_items": congestion_top_items,
            "accident_top_items": accident_top_items,
            "control_top_items": control_top_items,
        }

    @staticmethod
    def _string_or_placeholder(value: object | None) -> str:
        """把任意值转成可读字符串，缺失时返回占位符。"""

        if value is None:
            return "无"
        text = str(value).strip()
        return text if text else "无"

    @classmethod
    def _format_direction(cls, direction_type: object | None) -> str:
        """把方向编码转成自然语言。"""

        normalized = cls._string_or_placeholder(direction_type)
        direction_map = {
            "0": "双向",
            "1": "上行",
            "2": "下行",
        }
        return direction_map.get(normalized, normalized)

    @staticmethod
    def _format_milestone(value: object | None) -> str:
        """格式化桩号。"""

        if value is None:
            return "未知"
        if isinstance(value, str):
            text = value.strip()
            return text if text else "未知"
        return str(value)

    @staticmethod
    def _format_number(value: object | None) -> str:
        """格式化数值。"""

        if value is None:
            return "未知"
        if isinstance(value, str):
            text = value.strip()
            return text if text else "未知"
        return str(value)

    @classmethod
    def _build_report_event_line(cls, item: dict[str, object]) -> str:
        """把单条 topN 事件整理成一行模板文本。"""

        road_code = cls._string_or_placeholder(item.get("roadGBCode") or item.get("roadGbCode"))
        road_name = cls._string_or_placeholder(item.get("roadName"))
        direction = cls._format_direction(item.get("directionType"))
        begin = cls._format_milestone(item.get("beginMilestone"))
        end = cls._format_milestone(item.get("endMilestone"))
        road_amble_mile = cls._format_number(item.get("roadAmbleMile"))
        control_measures = cls._string_or_placeholder(item.get("controlMeasures"))
        situation_remark = cls._string_or_placeholder(item.get("situationRemark"))
        jeeves = cls._string_or_placeholder(item.get("jeeves"))
        begin_time = cls._string_or_placeholder(item.get("beginTime"))
        expected_end_time = cls._string_or_placeholder(
            item.get("expectedEndTime") or item.get("expectedTime") or item.get("endTime")
        )
        des = cls._string_or_placeholder(item.get("des"))
        return (
            f"- {road_code}，{road_name}，方向：{direction}，K{begin}~K{end}，"
            f"缓行约 {road_amble_mile} 公里，管制措施：{control_measures}，"
            f"现场情况备注：{situation_remark}，占道情况：{jeeves}，"
            f"开始时间：{begin_time}-预期结束时间：{expected_end_time}，事件描述：{des}"
        )

    @classmethod
    def _build_report_section(cls, title: str, items: list[dict[str, object]]) -> str:
        """把一类 topN 事件整理成分组文本。"""

        lines = [f"{title}："]
        if not items:
            lines.append("- 暂无")
            return "\n".join(lines)
        for item in items:
            lines.append(cls._build_report_event_line(item))
        return "\n".join(lines)

    @staticmethod
    def _extract_payload_items(
        response_payload: dict[str, object] | list[dict[str, object]],
        key: str,
    ) -> list[dict[str, object]]:
        """从响应中提取某类 topN 列表。"""

        if not isinstance(response_payload, dict):
            return []
        items = response_payload.get(key, [])
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    @classmethod
    def _build_compact_report_context(
        cls,
        *,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> str:
        """把整体路网结果整理成模板化报表上下文。"""

        query_time = "-"
        congestion_total_mile = None
        if isinstance(response_payload, dict):
            query_time = cls._string_or_placeholder(response_payload.get("queryTime"))
            congestion_payload = response_payload.get("congestion", {})
            if isinstance(congestion_payload, dict):
                congestion_total_mile = congestion_payload.get("totalMile")
        congestion_items = cls._extract_payload_items(response_payload, "congestionTopN")
        # accident_items = cls._extract_payload_items(response_payload, "accidentTopN")
        control_items = cls._extract_payload_items(response_payload, "controlTopN")

        lines = [f"查询时间：{query_time}"]
        if congestion_total_mile is not None:
            lines.append(f"拥堵总里程：{cls._format_number(congestion_total_mile)} 公里")
        lines.append(cls._build_report_section("拥堵列表", congestion_items))
        lines.append("")
        # lines.append(cls._build_report_section("事故列表", accident_items))
        # lines.append("")
        lines.append(cls._build_report_section("管制列表", control_items))
        return "\n".join(lines)

    @staticmethod
    def _build_report_context(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> str:
        """把结构化参数和接口返回转为报表类 system 上下文。"""

        del resolved_arguments
        return "\n".join(
            [
                REPORT_CONTEXT_PROMPT_PREFIX,
                ReportNode._build_compact_report_context(response_payload=response_payload),
            ]
        )
