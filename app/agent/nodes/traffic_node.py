"""路况业务节点模块。

负责把当前路况问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问实时路况接口。
"""

from __future__ import annotations

from json import dumps

from app.agent.prompts import TRAFFIC_CONTEXT_PROMPT_PREFIX
from app.agent.state import (
    AgentState,
    ExecutorResult,
    ResolvedArguments,
    merge_step_result,
    resolve_active_execution_step_id,
    resolve_step_arguments,
)


class TrafficNode:
    """LangGraph 路况业务节点。"""

    async def run(self, state: AgentState) -> dict[str, object]:
        """生成路况问题的业务上下文。"""

        step_id = resolve_active_execution_step_id(
            state,
            executor="traffic",
            default_step_id="traffic_1",
        )
        resolved_arguments = resolve_step_arguments(state, step_id=step_id, executor="traffic")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"traffic_context": None}
        executor_result = ExecutorResult(
            step_id=step_id,
            executor="traffic",
            is_success=True,
            raw_result=dict(resolved_arguments.arguments),
            normalized_result=dict(resolved_arguments.arguments),
            summary="已整理路况查询所需的结构化参数。",
        )
        return {
            "traffic_context": self._build_traffic_context(resolved_arguments),
            **merge_step_result(state, result=executor_result),
        }

    @staticmethod
    def _build_traffic_context(resolved_arguments: ResolvedArguments) -> str:
        """把结构化参数转为路况类 system 上下文。"""

        return "\n".join(
            [
                TRAFFIC_CONTEXT_PROMPT_PREFIX,
                dumps(resolved_arguments.arguments, ensure_ascii=False, indent=2),
            ]
        )
