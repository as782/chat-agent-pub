"""参数提取节点模块。

负责在 planner 输出主分类之后，生成当前问题的结构化参数结果。
当前阶段先提供规则式实现，为后续切换到 LLM argument resolver 预留统一接口。
"""

from __future__ import annotations

from app.agent.argument_resolver import ArgumentResolver
from app.agent.state import AgentState, ResolvedArguments
from app.core.logger import get_logger

LOGGER = get_logger(__name__)


class ArgumentNode:
    """LangGraph 参数提取节点。"""

    def __init__(self, *, argument_resolver: ArgumentResolver | None = None) -> None:
        self._argument_resolver = argument_resolver or ArgumentResolver()

    async def run(self, state: AgentState) -> dict[str, object]:
        """提取当前问题对应的结构化参数。"""

        LOGGER.info("========== 参数提取开始 ==========")
        
        resolved_arguments = self._argument_resolver.resolve(state)
        step_arguments = self._build_step_arguments(state)
        missing_fields = self._collect_missing_fields(
            resolved_arguments=resolved_arguments,
            step_arguments=step_arguments,
        )
        
        LOGGER.info(
            "参数提取完成：\n"
            "  主参数 category=%s, 缺失字段：%s\n"
            "  逐步参数数量：%s",
            resolved_arguments.category,
            list(resolved_arguments.missing_fields),
            len(step_arguments),
        )
        for step_id, args in step_arguments.items():
            LOGGER.info("    步骤 %s: category=%s, 缺失字段=%s", 
                       step_id, args.category, list(args.missing_fields))

        need_clarification = bool(missing_fields) or bool(state.get("need_clarification", False))
        clarification_question = state.get("clarification_question")
        if need_clarification and clarification_question is None and missing_fields:
            clarification_question = self._build_clarification_question(missing_fields)
            LOGGER.info("需要澄清：missing_fields=%s", missing_fields)

        LOGGER.info("================================\n")
        
        return {
            "resolved_arguments": resolved_arguments,
            "step_arguments": step_arguments,
            "need_clarification": need_clarification,
            "clarification_question": clarification_question,
        }

    def _build_step_arguments(self, state: AgentState) -> dict[str, ResolvedArguments]:
        """按 execution_plan 中的执行步骤生成逐步参数。"""

        execution_plan = state.get("execution_plan")
        if execution_plan is None:
            return {}

        step_arguments: dict[str, ResolvedArguments] = {}
        for step in execution_plan.steps:
            if step.executor == "answer":
                continue
            step_arguments[step.step_id] = self._argument_resolver.resolve_for_executor(
                state,
                executor=step.executor,
            )
        if step_arguments:
            return step_arguments

        fallback_step = self._build_fallback_step(state)
        if fallback_step is None:
            return {}
        step_arguments[fallback_step[0]] = fallback_step[1]
        return step_arguments

    def _build_fallback_step(
        self,
        state: AgentState,
    ) -> tuple[str, ResolvedArguments] | None:
        """在 execution_plan 未携带步骤时，为当前主分类生成最小兜底参数。"""

        primary_category = str(state.get("primary_category", "general"))
        if primary_category == "route_planning":
            return (
                "route_1",
                self._argument_resolver.resolve_for_executor(state, executor="route"),
            )
        if primary_category == "traffic_status":
            return (
                "traffic_1",
                self._argument_resolver.resolve_for_executor(state, executor="traffic"),
            )
        if primary_category == "network_report":
            return (
                "report_1",
                self._argument_resolver.resolve_for_executor(state, executor="report"),
            )
        if primary_category == "policy":
            return (
                "rag_1",
                self._argument_resolver.resolve_for_executor(state, executor="rag"),
            )
        if state.get("requested_tool_names"):
            return (
                "tool_1",
                self._argument_resolver.resolve_for_executor(state, executor="tool"),
            )
        return None

    @staticmethod
    def _collect_missing_fields(
        *,
        resolved_arguments: ResolvedArguments,
        step_arguments: dict[str, ResolvedArguments],
    ) -> list[str]:
        """汇总主参数和逐步参数里的缺失字段。"""

        merged_missing_fields = list(resolved_arguments.missing_fields)
        for step_resolved_arguments in step_arguments.values():
            for field_name in step_resolved_arguments.missing_fields:
                if field_name not in merged_missing_fields:
                    merged_missing_fields.append(field_name)
        return merged_missing_fields

    @staticmethod
    def _build_clarification_question(missing_fields: list[str]) -> str:
        """根据缺失字段生成最小澄清问题。"""

        field_labels = {
            "origin": "起点",
            "destination": "终点",
        }
        readable_fields = [field_labels.get(field, field) for field in missing_fields]
        return f"请补充以下信息后再继续：{'、'.join(readable_fields)}。"
