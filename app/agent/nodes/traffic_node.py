"""路况业务节点模块。

负责把当前路况问题的结构化参数整理为可注入回答节点的业务上下文。
当前阶段只做任务规范化，不直接访问实时路况接口。
"""

from __future__ import annotations

from json import dumps

from app.agent.prompts import TRAFFIC_CONTEXT_PROMPT_PREFIX
from app.agent.state import AgentState, ResolvedArguments


class TrafficNode:
    """LangGraph 路况业务节点。"""

    async def run(self, state: AgentState) -> dict[str, object]:
        """生成路况问题的业务上下文。"""

        resolved_arguments = state.get("resolved_arguments")
        if not isinstance(resolved_arguments, ResolvedArguments):
            return {"traffic_context": None}

        return {
            "traffic_context": self._build_traffic_context(resolved_arguments),
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
