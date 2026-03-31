"""服务区业务节点模块。

负责在服务区类问题中直接调用标准工具查询服务区、充电和商业配套信息，
并把结果整理为可注入回答节点的上下文。
"""

from __future__ import annotations

from json import dumps, loads

from app.agent.prompts import SERVICE_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
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
        query_arguments = self._build_tool_arguments(resolved_arguments)
        try:
            tool_output = await self._tool_registry.execute_named_tool(
                tool_name="live_service_query",
                arguments=query_arguments,
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
                    resolved_arguments=resolved_arguments,
                    response_payload=response_payload,
                ),
                summary=f"服务区查询成功，命中 {len(response_payload)} 条结果。",
            )
            return {
                "service_context": self._build_service_context(
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
        """把结构化参数转换为服务区查询工具参数。"""

        return {
            "keyword": str(
                resolved_arguments.arguments.get("keyword")
                or resolved_arguments.arguments.get("query")
                or ""
            )
        }

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
        resolved_arguments: ResolvedArguments,
        response_payload: list[dict[str, object]],
    ) -> dict[str, object]:
        """提取服务区查询结果中的关键摘要字段。"""

        first_result = response_payload[0] if response_payload else {}
        charge_list = first_result.get("chargeList", [])
        commercial_list = first_result.get("commercialList", [])
        tags = first_result.get("tags", [])
        return {
            "keyword": resolved_arguments.arguments.get("keyword"),
            "result_count": len(response_payload),
            "service_name": first_result.get("serviceName"),
            "road_name": first_result.get("roadName"),
            "status_tag": first_result.get("statusTag"),
            "charge_brand_count": len(charge_list) if isinstance(charge_list, list) else 0,
            "commercial_count": len(commercial_list) if isinstance(commercial_list, list) else 0,
            "tag_count": len(tags) if isinstance(tags, list) else 0,
        }

    @staticmethod
    def _build_service_context(
        *,
        resolved_arguments: ResolvedArguments,
        response_payload: list[dict[str, object]],
    ) -> str:
        """把结构化参数和接口返回转为服务区类 system 上下文。"""

        return "\n".join(
            [
                SERVICE_CONTEXT_PROMPT_PREFIX,
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
