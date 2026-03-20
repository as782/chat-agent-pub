"""路线规划业务节点模块。
负责把当前路线规划问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问外部路线接口，后续可在此节点内部切换为 HTTP 或 MCP 执行。
"""

from __future__ import annotations

from json import dumps

from app.agent.prompts import ROUTE_CONTEXT_PROMPT_PREFIX
from app.agent.state import (
    AgentState,
    ExecutorResult,
    ResolvedArguments,
    merge_step_result,
    resolve_active_execution_step_id,
    resolve_step_arguments,
)


class RouteNode:
    """LangGraph 路线规划业务节点。"""

    async def run(self, state: AgentState) -> dict[str, object]:
        """生成路线规划问题的业务上下文。"""

        step_id = resolve_active_execution_step_id(
            state,
            executor="route",
            default_step_id="route_1",
        )
        resolved_arguments = resolve_step_arguments(state, step_id=step_id, executor="route")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"route_context": None}
        executor_result = ExecutorResult(
            step_id=step_id,
            executor="route",
            is_success=True,
            raw_result=dict(resolved_arguments.arguments),
            normalized_result=dict(resolved_arguments.arguments),
            summary="已整理路线规划所需的结构化参数。",
        )
        return {
            "route_context": self._build_route_context(resolved_arguments),
            **merge_step_result(state, result=executor_result),
        }

    @staticmethod
    def _build_route_context(resolved_arguments: ResolvedArguments) -> str:
        """把结构化参数转为路线类 system 上下文。"""

        return "\n".join(
            [
                ROUTE_CONTEXT_PROMPT_PREFIX,
                dumps(resolved_arguments.arguments, ensure_ascii=False, indent=2),
            ]
        )
