"""路况业务节点模块。

负责把当前路况问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问实时路况接口。
"""

from __future__ import annotations

from json import dumps, loads

from app.agent.prompts import TRAFFIC_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
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
        query_arguments = self._build_tool_arguments(resolved_arguments)
        try:
            tool_output = await self._tool_registry.execute_named_tool(
                tool_name="live_road_event_query",
                arguments=query_arguments,
            )
            response_payload = self._parse_tool_output(tool_output)
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="traffic",
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
                "traffic_context": self._build_traffic_context(
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
        """把结构化参数转换为路况查询工具参数。"""

        road = str(
            resolved_arguments.arguments.get("road")
            or resolved_arguments.arguments.get("target")
            or resolved_arguments.arguments.get("query")
            or ""
        )
        return {"road": road}

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
        resolved_arguments: ResolvedArguments,
        response_payload: list[dict[str, object]],
    ) -> dict[str, object]:
        """提取路况查询结果中的关键摘要字段。"""

        first_road = response_payload[0] if response_payload else {}
        congestion_count = len(first_road.get("congestionInfoList", []))
        traffic_control_count = len(first_road.get("trafficControlList", []))
        service_area_count = len(first_road.get("serviceAreaList", []))
        exit_count = len(first_road.get("exitInfoList", []))
        return {
            "road": resolved_arguments.arguments.get("road")
            or resolved_arguments.arguments.get("target"),
            "road_name": first_road.get("roadName"),
            "result_count": len(response_payload),
            "congestion_count": congestion_count,
            "traffic_control_count": traffic_control_count,
            "service_area_count": service_area_count,
            "exit_count": exit_count,
        }

    @staticmethod
    def _build_success_summary(response_payload: list[dict[str, object]]) -> str:
        """生成路况查询成功摘要。"""

        return f"路况查询成功，命中 {len(response_payload)} 条道路结果。"

    @staticmethod
    def _build_traffic_context(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: list[dict[str, object]],
    ) -> str:
        """把结构化参数和接口返回转为路况类 system 上下文。"""

        return "\n".join(
            [
                TRAFFIC_CONTEXT_PROMPT_PREFIX,
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
