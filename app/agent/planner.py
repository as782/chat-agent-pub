"""Agent 规划器模块。
负责根据用户问题给出业务分类和最小执行计划。
当前同时支持规则式规划和可选的 LLM planner，并统一输出 ExecutionPlan。
"""

from __future__ import annotations

from json import JSONDecodeError, loads
from re import DOTALL, search
from typing import Any

from app.agent.prompts import PLANNER_JSON_OUTPUT_PROMPT, PLANNER_PROMPT
from app.agent.state import (
    AgentRoute,
    AgentState,
    ExecutionPlan,
    ExecutionStep,
    ExecutorType,
    ProblemCategory,
)
from app.clients.llm_client import LlmChatCompletionResult, LlmClient, LlmInputMessage
from app.core.config import Settings, get_settings
from app.core.logger import get_logger

LOGGER = get_logger(__name__)

POLICY_KEYWORDS = ("政策", "标准", "规范", "制度", "规定", "依据", "口径")
ROUTE_KEYWORDS = ("怎么走", "路线", "导航", "到", "路径", "方案")
TRAFFIC_KEYWORDS = ("路况", "拥堵", "封闭", "施工", "事故", "缓行", "通行")
NETWORK_REPORT_KEYWORDS = ("全路网", "日报", "周报", "月报", "汇总", "表格", "对比", "分析")

_VALID_PROBLEM_CATEGORIES: set[ProblemCategory] = {
    "policy",
    "route_planning",
    "traffic_status",
    "network_report",
    "general",
}
_VALID_EXECUTORS: set[ExecutorType] = {
    "answer",
    "rag",
    "mcp",
    "tool",
    "route",
    "traffic",
    "report",
}


class PlannerService:
    """规划器服务。

    默认使用规则式规划，只有在显式开启配置时才调用 LLM planner。
    这样可以先接入更强的规划能力，同时保留当前稳定行为。
    """

    def __init__(
        self,
        *,
        llm_client: LlmClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._settings = settings or get_settings()

    async def build_plan_async(self, state: AgentState) -> ExecutionPlan:
        """根据当前状态生成分类与执行计划。"""

        if not self._should_use_llm_planner(state):
            return self.build_plan(state)

        try:
            return await self._build_plan_with_llm(state)
        except Exception as exception:  # noqa: BLE001
            LOGGER.warning(
                "LLM planner 规划失败，已回退到规则规划：error=%s",
                str(exception),
            )
            return self.build_plan(state)

    def build_plan(self, state: AgentState) -> ExecutionPlan:
        """使用规则逻辑生成分类与执行计划。"""

        latest_user_message = str(state.get("latest_user_message", ""))
        normalized_message = latest_user_message.lower()
        requested_tool_names = state.get("requested_tool_names") or []

        primary_category = self._detect_primary_category(
            latest_user_message=latest_user_message,
            normalized_message=normalized_message,
            has_requested_tools=bool(requested_tool_names),
        )
        recommended_route = self._build_recommended_route(
            primary_category=primary_category,
            has_requested_tools=bool(requested_tool_names),
        )
        steps = self._build_steps(
            primary_category=primary_category,
            has_requested_tools=bool(requested_tool_names),
            latest_user_message=latest_user_message,
            normalized_message=normalized_message,
        )
        execution_mode = self._resolve_execution_mode(steps)

        return ExecutionPlan(
            primary_category=primary_category,
            execution_mode=execution_mode,
            recommended_route=recommended_route,
            steps=steps,
        )

    def _should_use_llm_planner(self, state: AgentState) -> bool:
        """判断是否启用 LLM planner。"""

        if state.get("requested_tool_names"):
            return False
        if not self._settings.planner_use_llm:
            return False
        return self._llm_client is not None

    async def _build_plan_with_llm(self, state: AgentState) -> ExecutionPlan:
        """调用 LLM 生成规划结果。"""

        if self._llm_client is None:
            raise RuntimeError("LLM planner 未注入 llm_client。")

        completion_result = await self._llm_client.create_chat_completion(
            messages=self._build_planner_messages(state),
            model_name=self._settings.planner_model or state.get("model_name"),
            enable_thinking=False,
        )
        return self._parse_llm_plan(state, completion_result)

    def _build_planner_messages(self, state: AgentState) -> list[LlmInputMessage]:
        """构造 planner LLM 的输入消息。"""

        latest_user_message = str(state.get("latest_user_message", "")).strip()
        input_messages = state.get("input_messages") or []
        formatted_history_lines: list[str] = []
        for message in input_messages[-6:]:
            if not isinstance(message, LlmInputMessage):
                continue
            formatted_history_lines.append(f"- {message.role}: {message.content}")

        user_prompt_lines = [
            "请根据下面的用户问题生成分类与执行计划。",
            f"latest_user_message: {latest_user_message}",
        ]
        if formatted_history_lines:
            user_prompt_lines.append("recent_messages:")
            user_prompt_lines.extend(formatted_history_lines)

        return [
            LlmInputMessage(
                role="system",
                content=f"{PLANNER_PROMPT}\n\n{PLANNER_JSON_OUTPUT_PROMPT}",
            ),
            LlmInputMessage(role="user", content="\n".join(user_prompt_lines)),
        ]

    def _parse_llm_plan(
        self,
        state: AgentState,
        completion_result: LlmChatCompletionResult,
    ) -> ExecutionPlan:
        """解析 LLM planner 的结构化结果。"""

        payload = self._extract_json_payload(completion_result.content)
        requested_tool_names = state.get("requested_tool_names") or []
        latest_user_message = str(state.get("latest_user_message", ""))
        normalized_message = latest_user_message.lower()

        primary_category = self._coerce_primary_category(payload.get("primary_category"))
        if primary_category is None:
            primary_category = self._detect_primary_category(
                latest_user_message=latest_user_message,
                normalized_message=normalized_message,
                has_requested_tools=bool(requested_tool_names),
            )

        fallback_steps = self._build_steps(
            primary_category=primary_category,
            has_requested_tools=bool(requested_tool_names),
            latest_user_message=latest_user_message,
            normalized_message=normalized_message,
        )
        steps = self._coerce_steps(payload.get("steps"), fallback_steps=fallback_steps)
        recommended_route = self._derive_recommended_route(
            steps=steps,
            primary_category=primary_category,
            has_requested_tools=bool(requested_tool_names),
        )

        clarification_question = payload.get("clarification_question")
        if clarification_question is not None and not isinstance(clarification_question, str):
            clarification_question = None

        return ExecutionPlan(
            primary_category=primary_category,
            execution_mode=self._resolve_execution_mode(steps),
            recommended_route=recommended_route,
            need_clarification=self._coerce_bool(payload.get("need_clarification")),
            clarification_question=clarification_question,
            steps=steps,
        )

    @staticmethod
    def _extract_json_payload(raw_content: str) -> dict[str, Any]:
        """从 LLM 文本结果中提取 JSON 对象。"""

        stripped_content = raw_content.strip()
        fenced_match = search(r"```(?:json)?\s*(\{.*\})\s*```", stripped_content, DOTALL)
        if fenced_match:
            candidate_text = fenced_match.group(1)
        else:
            start_index = stripped_content.find("{")
            end_index = stripped_content.rfind("}")
            if start_index == -1 or end_index == -1 or end_index <= start_index:
                raise ValueError("LLM planner 未返回合法 JSON 对象。")
            candidate_text = stripped_content[start_index : end_index + 1]

        try:
            payload = loads(candidate_text)
        except JSONDecodeError as exception:
            raise ValueError("LLM planner 返回的 JSON 无法解析。") from exception

        if not isinstance(payload, dict):
            raise ValueError("LLM planner 返回结果必须是 JSON 对象。")
        return payload

    @staticmethod
    def _coerce_primary_category(value: object) -> ProblemCategory | None:
        """校验并规范化主分类。"""

        if isinstance(value, str) and value in _VALID_PROBLEM_CATEGORIES:
            return value
        return None

    def _coerce_steps(
        self,
        value: object,
        *,
        fallback_steps: list[ExecutionStep],
    ) -> list[ExecutionStep]:
        """把 LLM 输出的 steps 规整为统一步骤对象。"""

        if not isinstance(value, list):
            return fallback_steps

        steps: list[ExecutionStep] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue

            executor = self._coerce_executor(item.get("executor"))
            if executor is None:
                continue

            step_id = item.get("step_id")
            goal = item.get("goal")
            if not isinstance(step_id, str) or not step_id:
                step_id = f"{executor}_{index}"
            if not isinstance(goal, str) or not goal.strip():
                goal = f"执行 {executor} 步骤"

            depends_on = item.get("depends_on")
            normalized_depends_on = (
                [value for value in depends_on if isinstance(value, str)]
                if isinstance(depends_on, list)
                else []
            )
            metadata = item.get("metadata")
            normalized_metadata = metadata if isinstance(metadata, dict) else {}

            steps.append(
                ExecutionStep(
                    step_id=step_id,
                    executor=executor,
                    goal=goal,
                    depends_on=normalized_depends_on,
                    can_run_in_parallel=self._coerce_bool(item.get("can_run_in_parallel")),
                    metadata=normalized_metadata,
                )
            )

        if not steps:
            return fallback_steps

        if all(step.executor != "answer" for step in steps):
            answer_dependencies = [step.step_id for step in steps]
            steps.append(
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="根据已有执行结果生成最终回答",
                    depends_on=answer_dependencies,
                )
            )
        return steps

    @staticmethod
    def _coerce_executor(value: object) -> ExecutorType | None:
        """校验并规范化执行器类型。"""

        if isinstance(value, str) and value in _VALID_EXECUTORS:
            return value
        return None

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        """把常见布尔表达规整为 bool。"""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return False

    def _detect_primary_category(
        self,
        *,
        latest_user_message: str,
        normalized_message: str,
        has_requested_tools: bool,
    ) -> ProblemCategory:
        """识别当前问题的主业务分类。"""

        has_route_request = normalized_message.startswith("mcp:") or any(
            keyword in latest_user_message for keyword in ROUTE_KEYWORDS
        )
        if (
            latest_user_message.startswith("知识库:")
            or normalized_message.startswith(("knowledge:", "konwledge:"))
            or "#knowledge" in normalized_message
        ):
            return "policy"
        if any(keyword in latest_user_message for keyword in NETWORK_REPORT_KEYWORDS):
            return "network_report"
        if normalized_message.startswith("mcp:") or "#mcp" in normalized_message:
            return "route_planning"
        if has_route_request:
            return "route_planning"
        if any(keyword in latest_user_message for keyword in TRAFFIC_KEYWORDS):
            return "traffic_status"
        if any(keyword in latest_user_message for keyword in POLICY_KEYWORDS):
            return "policy"
        if has_requested_tools:
            return "general"
        return "general"

    @staticmethod
    def _build_recommended_route(
        *,
        primary_category: ProblemCategory,
        has_requested_tools: bool,
    ) -> AgentRoute:
        """给出当前计划建议的技术路由。"""

        if has_requested_tools:
            return "tool"
        if primary_category == "policy":
            return "ragflow"
        if primary_category == "route_planning":
            return "route"
        if primary_category == "traffic_status":
            return "traffic"
        if primary_category == "network_report":
            return "report"
        return "answer"

    def _derive_recommended_route(
        self,
        *,
        steps: list[ExecutionStep],
        primary_category: ProblemCategory,
        has_requested_tools: bool,
    ) -> AgentRoute:
        """根据首个非 answer 步骤推导推荐路由。"""

        if has_requested_tools:
            return "tool"

        for step in steps:
            if step.executor == "answer":
                continue
            if step.executor == "rag":
                return "ragflow"
            if step.executor in {"route", "traffic", "report", "tool", "mcp"}:
                return step.executor
        return self._build_recommended_route(
            primary_category=primary_category,
            has_requested_tools=has_requested_tools,
        )

    @staticmethod
    def _resolve_execution_mode(steps: list[ExecutionStep]) -> str:
        """根据步骤数量推导执行模式。"""

        execution_mode = "direct" if len(steps) <= 1 else "single_step"
        if len(steps) > 2:
            execution_mode = "multi_step"
        return execution_mode

    @staticmethod
    def _build_steps(
        *,
        primary_category: ProblemCategory,
        has_requested_tools: bool,
        latest_user_message: str,
        normalized_message: str,
    ) -> list[ExecutionStep]:
        """根据主分类生成最小可执行步骤。"""

        if has_requested_tools:
            return [
                ExecutionStep(
                    step_id="tool_1",
                    executor="tool",
                    goal="执行用户显式开放的工具",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="根据工具结果生成最终回答",
                    depends_on=["tool_1"],
                ),
            ]

        if primary_category == "policy":
            return [
                ExecutionStep(
                    step_id="rag_1",
                    executor="rag",
                    goal="检索政策和标准相关知识",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结政策检索结果并回答用户",
                    depends_on=["rag_1"],
                ),
            ]

        if primary_category == "route_planning":
            need_policy_support = _needs_policy_support_for_route(
                latest_user_message=latest_user_message,
                normalized_message=normalized_message,
            )
            need_traffic_support = _needs_traffic_support_for_route(
                latest_user_message=latest_user_message,
            )
            route_dependency_step_ids: list[str] = []
            route_steps: list[ExecutionStep] = []

            if need_policy_support:
                route_steps.append(
                    ExecutionStep(
                        step_id="rag_1",
                        executor="rag",
                        goal="检索路线相关政策和标准要求",
                        can_run_in_parallel=True,
                    )
                )
                route_dependency_step_ids.append("rag_1")

            route_steps.append(
                ExecutionStep(
                    step_id="route_1",
                    executor="route",
                    goal="查询路线规划相关数据",
                    can_run_in_parallel=True,
                )
            )
            route_dependency_step_ids.append("route_1")

            if need_traffic_support:
                route_steps.append(
                    ExecutionStep(
                        step_id="traffic_1",
                        executor="traffic",
                        goal="查询路线相关路况信息",
                        can_run_in_parallel=True,
                    )
                )
                route_dependency_step_ids.append("traffic_1")

            if len(route_dependency_step_ids) > 1:
                route_steps.append(
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="结合路线、路况和政策结果生成最终回答",
                        depends_on=route_dependency_step_ids,
                    )
                )
                return route_steps
            return [
                ExecutionStep(
                    step_id="route_1",
                    executor="route",
                    goal="查询路线规划相关数据",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结路线结果并回答用户",
                    depends_on=["route_1"],
                ),
            ]

        if primary_category == "traffic_status":
            if any(keyword in latest_user_message for keyword in POLICY_KEYWORDS):
                return [
                    ExecutionStep(
                        step_id="rag_1",
                        executor="rag",
                        goal="检索路况研判相关政策和标准要求",
                        can_run_in_parallel=True,
                    ),
                    ExecutionStep(
                        step_id="traffic_1",
                        executor="traffic",
                        goal="查询路况或实时交通数据",
                        can_run_in_parallel=True,
                    ),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="结合路况结果与政策要求生成最终回答",
                        depends_on=["rag_1", "traffic_1"],
                    ),
                ]
            return [
                ExecutionStep(
                    step_id="traffic_1",
                    executor="traffic",
                    goal="查询路况或实时交通数据",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结路况结果并回答用户",
                    depends_on=["traffic_1"],
                ),
            ]

        if primary_category == "network_report":
            if any(keyword in latest_user_message for keyword in POLICY_KEYWORDS):
                return [
                    ExecutionStep(
                        step_id="rag_1",
                        executor="rag",
                        goal="检索路网报告相关政策和标准要求",
                        can_run_in_parallel=True,
                    ),
                    ExecutionStep(
                        step_id="report_1",
                        executor="report",
                        goal="汇总多个区域或多个接口的路网数据",
                        can_run_in_parallel=True,
                    ),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="结合路网数据与政策要求输出报告、对比结论和表格",
                        depends_on=["rag_1", "report_1"],
                    ),
                ]
            return [
                ExecutionStep(
                    step_id="report_1",
                    executor="report",
                    goal="汇总多个区域或多个接口的路网数据",
                    can_run_in_parallel=True,
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="输出路网报告、对比结论和表格",
                    depends_on=["report_1"],
                ),
            ]

        return [
            ExecutionStep(
                step_id="answer_1",
                executor="answer",
                goal="直接回答用户问题",
            )
        ]


def _needs_policy_support_for_route(
    *,
    latest_user_message: str,
    normalized_message: str,
) -> bool:
    """判断路线类问题是否同时需要补充政策标准检索。"""

    return (
        any(keyword in latest_user_message for keyword in POLICY_KEYWORDS)
        or latest_user_message.startswith("知识库:")
        or normalized_message.startswith(("knowledge:", "konwledge:"))
        or "#knowledge" in normalized_message
    )


def _needs_traffic_support_for_route(*, latest_user_message: str) -> bool:
    """判断路线类问题是否同时需要补充路况信息。"""

    return any(keyword in latest_user_message for keyword in TRAFFIC_KEYWORDS)
