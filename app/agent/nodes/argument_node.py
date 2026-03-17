"""参数提取节点模块。

负责在 planner 输出主分类之后，生成当前问题的结构化参数结果。
当前阶段先提供规则式实现，为后续切换到 LLM argument resolver 预留统一接口。
"""

from __future__ import annotations

from app.agent.argument_resolver import ArgumentResolver
from app.agent.state import AgentState


class ArgumentNode:
    """LangGraph 参数提取节点。"""

    def __init__(self, *, argument_resolver: ArgumentResolver | None = None) -> None:
        self._argument_resolver = argument_resolver or ArgumentResolver()

    async def run(self, state: AgentState) -> dict[str, object]:
        """提取当前问题对应的结构化参数。"""

        resolved_arguments = self._argument_resolver.resolve(state)
        need_clarification = bool(resolved_arguments.missing_fields) or bool(
            state.get("need_clarification", False)
        )
        clarification_question = state.get("clarification_question")
        if (
            need_clarification
            and clarification_question is None
            and resolved_arguments.missing_fields
        ):
            clarification_question = self._build_clarification_question(
                resolved_arguments.missing_fields
            )

        return {
            "resolved_arguments": resolved_arguments,
            "need_clarification": need_clarification,
            "clarification_question": clarification_question,
        }

    @staticmethod
    def _build_clarification_question(missing_fields: list[str]) -> str:
        """根据缺失字段生成最小澄清问题。"""

        field_labels = {
            "origin": "起点",
            "destination": "终点",
        }
        readable_fields = [field_labels.get(field, field) for field in missing_fields]
        return f"请补充以下信息后再继续：{'、'.join(readable_fields)}。"
