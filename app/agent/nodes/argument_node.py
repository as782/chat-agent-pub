"""参数提取节点模块。

负责在 planner 输出主分类之后，生成当前问题的结构化参数结果。
当前阶段先提供规则式实现，为后续切换到 LLM argument resolver 预留统一接口。
"""

from __future__ import annotations

from app.agent.argument_resolver import ArgumentResolver
from app.agent.state import AgentState, ExecutionStep, ResolvedArguments


class ArgumentNode:
    """LangGraph 参数提取节点。"""

    def __init__(self, *, argument_resolver: ArgumentResolver | None = None) -> None:
        self._argument_resolver = argument_resolver or ArgumentResolver()

    async def run(self, state: AgentState) -> dict[str, object]:
        """提取当前问题对应的结构化参数。"""

        resolved_arguments = self._argument_resolver.resolve(state)
        step_arguments = self._build_step_arguments(state)
        missing_fields = self._collect_missing_fields(
            resolved_arguments=resolved_arguments,
            step_arguments=step_arguments,
        )
        need_clarification = bool(missing_fields) or bool(state.get("need_clarification", False))
        clarification_question = state.get("clarification_question")
        if need_clarification and clarification_question is None and missing_fields:
            clarification_question = self._build_clarification_question(missing_fields)

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
            resolved_arguments = self._argument_resolver.resolve_for_executor(
                state,
                executor=step.executor,
            )
            step_arguments[step.step_id] = self._merge_step_metadata(
                resolved_arguments=resolved_arguments,
                step=step,
            )
        if step_arguments:
            return step_arguments

        fallback_step = self._build_fallback_step(state)
        if fallback_step is None:
            return {}
        step_arguments[fallback_step[0]] = fallback_step[1]
        return step_arguments

    @staticmethod
    def _merge_step_metadata(
        *,
        resolved_arguments: ResolvedArguments,
        step: ExecutionStep,
    ) -> ResolvedArguments:
        """Use planner metadata as the primary source and keep resolver values as fallback."""

        if not step.metadata:
            return resolved_arguments

        merged_arguments = dict(resolved_arguments.arguments)
        for key, value in step.metadata.items():
            if not ArgumentNode._is_empty_metadata_value(value):
                merged_arguments[key] = value

        extraction_mode = resolved_arguments.extraction_mode
        if "planner_metadata" not in extraction_mode:
            extraction_mode = f"{extraction_mode}+planner_metadata"

        return ResolvedArguments(
            category=resolved_arguments.category,
            arguments=merged_arguments,
            missing_fields=list(resolved_arguments.missing_fields),
            extraction_mode=extraction_mode,
        )

    @staticmethod
    def _is_empty_metadata_value(value: object) -> bool:
        """判断 metadata 字段是否为空。"""

        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

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
        if primary_category == "service_area":
            return (
                "service_1",
                self._argument_resolver.resolve_for_executor(state, executor="service"),
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
