"""路网报告业务节点模块。

负责把路网汇总和报表类问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做报表任务规范化，不直接访问外部数据接口。
"""

from __future__ import annotations

from json import dumps, loads

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
                    resolved_arguments=resolved_arguments,
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
        resolved_arguments: ResolvedArguments,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> dict[str, object]:
        """提取整体路网查询结果中的关键摘要字段。"""

        if isinstance(response_payload, list):
            record_count = len(response_payload)
            congestion_total_mile = None
            query_time = None
            congestion_top_count = 0
            accident_top_count = 0
            control_top_count = 0
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
            record_count = 1
        return {
            "scope": resolved_arguments.arguments.get("scope"),
            "report_type": resolved_arguments.arguments.get("report_type"),
            "need_table": resolved_arguments.arguments.get("need_table"),
            "need_comparison": resolved_arguments.arguments.get("need_comparison"),
            "record_count": record_count,
            "query_time": query_time,
            "congestion_total_mile": congestion_total_mile,
            "congestion_top_count": congestion_top_count,
            "accident_top_count": accident_top_count,
            "control_top_count": control_top_count,
        }

    @staticmethod
    def _build_report_context(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: dict[str, object] | list[dict[str, object]],
    ) -> str:
        """把结构化参数和接口返回转为报表类 system 上下文。"""

        return "\n".join(
            [
                REPORT_CONTEXT_PROMPT_PREFIX,
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
