"""路线规划业务节点模块。
负责把当前路线规划问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问外部路线接口，后续可在此节点内部切换为 HTTP 或 MCP 执行。
"""

from __future__ import annotations

from json import dumps, loads

from app.agent.prompts import ROUTE_CONTEXT_PROMPT_PREFIX
from app.agent.state import (
    AgentState,
    ExecutorResult,
    ResolvedArguments,
    merge_step_result,
    resolve_active_execution_step_id,
    resolve_step_arguments,
)
from app.core.exceptions import AppException
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
        except AppException as exception:
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="route",
                is_success=False,
                raw_result={"query_arguments": dict(query_arguments)},
                normalized_result=dict(query_arguments),
                summary="路线查询失败。",
                error=exception.message,
            )
            return {
                "route_context": self._build_error_context(
                    resolved_arguments=resolved_arguments,
                    query_arguments=query_arguments,
                    error_message=exception.message,
                ),
                **merge_step_result(state, result=executor_result),
            }

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
        """提取路线查询结果中的关键摘要字段。"""

        routes = response_payload.get("routes", [])
        first_route = routes[0] if isinstance(routes, list) and routes else {}
        if not isinstance(first_route, dict):
            first_route = {}
        sections = first_route.get("sections", [])
        if not isinstance(sections, list):
            sections = []
        traffic_control_count = sum(
            len(section.get("trafficControls", []))
            for section in sections
            if isinstance(section, dict) and isinstance(section.get("trafficControls"), list)
        )
        service_area_count = sum(
            len(section.get("serviceAreas", []))
            for section in sections
            if isinstance(section, dict) and isinstance(section.get("serviceAreas"), list)
        )
        return {
            "origin": resolved_arguments.arguments.get("origin"),
            "destination": resolved_arguments.arguments.get("destination"),
            "travel_mode": resolved_arguments.arguments.get("travel_mode"),
            "routes_count": response_payload.get("routesCount")
            if response_payload.get("routesCount") is not None
            else len(routes) if isinstance(routes, list) else 0,
            "first_route_distance": first_route.get("distance"),
            "first_route_duration": first_route.get("duration"),
            "first_route_toll": first_route.get("toll"),
            "traffic_control_count": traffic_control_count,
            "service_area_count": service_area_count,
        }

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

    @staticmethod
    def _build_error_context(
        *,
        resolved_arguments: ResolvedArguments,
        query_arguments: dict[str, object],
        error_message: str,
    ) -> str:
        """构造路线查询失败时的上下文。"""

        return "\n".join(
            [
                ROUTE_CONTEXT_PROMPT_PREFIX,
                dumps(
                    {
                        "query_arguments": dict(query_arguments),
                        "resolved_arguments": dict(resolved_arguments.arguments),
                        "error": error_message,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
