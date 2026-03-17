"""路网报告业务节点模块。

负责把路网汇总和报表类问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做报表任务规范化，不直接访问外部数据接口。
"""

from __future__ import annotations

from json import dumps

from app.agent.prompts import REPORT_CONTEXT_PROMPT_PREFIX
from app.agent.state import (
    AgentState,
    ExecutorResult,
    ResolvedArguments,
    merge_step_result,
    resolve_execution_step_id,
    resolve_step_arguments,
)


class ReportNode:
    """LangGraph 路网报告业务节点。"""

    async def run(self, state: AgentState) -> dict[str, object]:
        """生成路网报告问题的业务上下文。"""

        step_id = resolve_execution_step_id(
            state,
            executor="report",
            default_step_id="report_1",
        )
        resolved_arguments = resolve_step_arguments(state, step_id=step_id, executor="report")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"report_context": None}
        executor_result = ExecutorResult(
            step_id=step_id,
            executor="report",
            is_success=True,
            raw_result=dict(resolved_arguments.arguments),
            normalized_result=dict(resolved_arguments.arguments),
            summary="已整理路网报告任务的结构化参数。",
        )
        return {
            "report_context": self._build_report_context(resolved_arguments),
            **merge_step_result(state, result=executor_result),
        }

    @staticmethod
    def _build_report_context(resolved_arguments: ResolvedArguments) -> str:
        """把结构化参数转为报表类 system 上下文。"""

        return "\n".join(
            [
                REPORT_CONTEXT_PROMPT_PREFIX,
                dumps(resolved_arguments.arguments, ensure_ascii=False, indent=2),
            ]
        )
