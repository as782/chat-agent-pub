"""参数提取节点模块。

负责在 planner 输出主分类之后，生成当前问题的结构化参数结果。
当前阶段先提供规则式实现，为后续切换到 LLM argument resolver 预留统一接口。
"""

from __future__ import annotations

from app.agent.argument_resolver import ArgumentResolver
from app.agent.od import validate_route_arguments
from app.agent.od.normalizer import clean_endpoint
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
                if step.executor == "route" and key in {"origin", "destination"}:
                    continue
                merged_arguments[key] = value
        route_missing_fields: list[str] | None = None
        if step.executor == "route":
            merged_arguments = ArgumentNode._merge_route_endpoint_metadata(
                merged_arguments=merged_arguments,
                planner_metadata=step.metadata,
            )
            route_missing_fields = ArgumentNode._collect_route_missing_fields(merged_arguments)
        if step.executor == "traffic":
            merged_arguments = ArgumentNode._normalize_traffic_road_arguments_after_merge(
                merged_arguments=merged_arguments,
                planner_metadata=step.metadata,
            )

        extraction_mode = resolved_arguments.extraction_mode
        if "planner_metadata" not in extraction_mode:
            extraction_mode = f"{extraction_mode}+planner_metadata"

        return ResolvedArguments(
            category=resolved_arguments.category,
            arguments=merged_arguments,
            missing_fields=(
                route_missing_fields
                if route_missing_fields is not None
                else list(resolved_arguments.missing_fields)
            ),
            extraction_mode=extraction_mode,
        )

    @staticmethod
    def _merge_route_endpoint_metadata(
        *,
        merged_arguments: dict[str, object],
        planner_metadata: dict[str, object],
    ) -> dict[str, object]:
        """Keep resolver OD values unless planner metadata is the only clean source."""

        normalized_arguments = dict(merged_arguments)
        resolver_valid, _ = validate_route_arguments(
            normalized_arguments.get("origin"),
            normalized_arguments.get("destination"),
        )
        planner_origin = clean_endpoint(planner_metadata.get("origin"))
        planner_destination = clean_endpoint(planner_metadata.get("destination"))
        planner_valid, _ = validate_route_arguments(planner_origin, planner_destination)
        if not resolver_valid and planner_valid:
            normalized_arguments["origin"] = planner_origin
            normalized_arguments["destination"] = planner_destination
            missing_fields = set()
        elif planner_valid:
            ArgumentNode._prefer_shorter_planner_endpoint(
                arguments=normalized_arguments,
                field_name="origin",
                planner_value=planner_origin,
            )
            ArgumentNode._prefer_shorter_planner_endpoint(
                arguments=normalized_arguments,
                field_name="destination",
                planner_value=planner_destination,
            )
            missing_fields = set()
        else:
            missing_fields = set()
            if not validate_route_arguments(normalized_arguments.get("origin"), "占位终点")[0]:
                missing_fields.add("origin")
            if not validate_route_arguments("占位起点", normalized_arguments.get("destination"))[0]:
                missing_fields.add("destination")
        if missing_fields:
            normalized_arguments["od_warnings"] = sorted(missing_fields)
        return normalized_arguments

    @staticmethod
    def _prefer_shorter_planner_endpoint(
        *,
        arguments: dict[str, object],
        field_name: str,
        planner_value: str,
    ) -> None:
        """用 planner 的更短干净端点修正 resolver 的结构文本噪声尾巴。

        resolver 允许未知地点，因此“上海正么样”这类城市后接口语噪声的片段
        可能看起来仍是合法地点。若 planner 同时给出“上海”这种更短前缀，
        这里优先采用更短地点短语。
        """

        current_value = str(arguments.get(field_name) or "").strip()
        if not current_value or not planner_value:
            return
        if current_value == planner_value:
            return
        if current_value.startswith(planner_value) and len(planner_value) >= 2:
            arguments[field_name] = planner_value

    @staticmethod
    def _collect_route_missing_fields(arguments: dict[str, object]) -> list[str]:
        """Collect missing route endpoints after resolver/planner metadata merge."""

        missing_fields: list[str] = []
        if not validate_route_arguments(arguments.get("origin"), "占位终点")[0]:
            missing_fields.append("origin")
        if not validate_route_arguments("占位起点", arguments.get("destination"))[0]:
            missing_fields.append("destination")
        if (
            arguments.get("origin") == arguments.get("destination")
            and "destination" not in missing_fields
        ):
            missing_fields.append("destination")
        return missing_fields

    @staticmethod
    def _normalize_traffic_road_arguments_after_merge(
        *,
        merged_arguments: dict[str, object],
        planner_metadata: dict[str, object],
    ) -> dict[str, object]:
        """Prefer planner canonical single-road fields over resolver surface roads."""

        normalized_arguments = dict(merged_arguments)
        has_planner_single_road = any(
            not ArgumentNode._is_empty_metadata_value(planner_metadata.get(field_name))
            for field_name in ("road", "road_name", "road_code")
        )
        has_planner_multi_roads = not ArgumentNode._is_empty_metadata_value(
            planner_metadata.get("roads")
        )
        if has_planner_single_road and not has_planner_multi_roads:
            normalized_arguments.pop("roads", None)
        return normalized_arguments

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
