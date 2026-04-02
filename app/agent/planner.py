"""Agent 规划器模块。
负责根据用户问题给出业务分类和最小执行计划。
当前同时支持规则式规划和可选的 LLM planner，并统一输出 ExecutionPlan。
"""

from __future__ import annotations

from json import JSONDecodeError, loads
from re import DOTALL, search
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.prompts import PLANNER_JSON_OUTPUT_PROMPT, PLANNER_PROMPT
from app.agent.state import (
    AgentRoute,
    AgentState,
    ExecutionPlan,
    ExecutionStep,
    ExecutorType,
    ProblemCategory,
)
from app.clients.llm_client import LlmClient, LlmInputMessage
from app.core.config import Settings, get_settings
from app.core.logger import get_logger

LOGGER = get_logger(__name__)

_VALID_PROBLEM_CATEGORIES: set[ProblemCategory] = {
    "policy",
    "route_planning",
    "traffic_status",
    "service_area",
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
    "service",
    "report",
}


class PlannerService:
    """规划器服务。
    使用LLM 对问题进行分类和规划。
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
        try:
            return await self._build_plan_with_llm(state)
        except Exception as exception:  # noqa: BLE001
            LOGGER.warning(
                "LLM planner 规划失败, error=%s",
                str(exception)
            )
            return self._build_fallback_plan(state)

    async def _build_plan_with_llm(self, state: AgentState) -> ExecutionPlan:
        """调用 LLM 生成规划结果。"""

        if self._llm_client is None:
            raise RuntimeError("LLM planner 未注入 llm_client。")

        planner_api_key = self._settings.planner_api_key
        completion_result = await self._llm_client.create_chat_completion(
            messages=self._build_planner_messages(state),
            model_name=self._settings.planner_model or state.get("model_name"),
            base_url=self._settings.planner_base_url or self._settings.openai_base_url,
            api_key=(
                planner_api_key.get_secret_value().strip() or None
                if planner_api_key is not None
                else None
            ),
            enable_thinking=False,
        )
        return self._parse_llm_plan(state, completion_result)

    def _build_planner_messages(self, state: AgentState) -> list[LlmInputMessage]:
        """构造 planner LLM 的输入消息。"""

        latest_user_message = str(state.get("latest_user_message", "")).strip()
        # 当前注释相关历史对话注入，只专注用户的最新输入问题来判断问题分类。
        # ====== 后续需要再添加 ======
        # input_messages = state.get("input_messages") or []
        # formatted_history_lines: list[str] = []
        # for message in input_messages[-6:]:
        #     if not isinstance(message, LlmInputMessage):
        #         continue
        #     formatted_history_lines.append(f"- {message.role}: {message.content}")

        # user_prompt_lines = [
        #     "请根据下面的用户问题生成分类与执行计划。",
        #     f"latest_user_message: {latest_user_message}",
        # ]
        # if formatted_history_lines:
        #     user_prompt_lines.append("recent_messages:")
        #     user_prompt_lines.extend(formatted_history_lines)
        # ====== 后续需要再添加 ======

        user_prompt_lines = [
            "请根据下面的用户问题生成分类与执行计划。",
            f"latest_user_message: {latest_user_message}",
        ]
        
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
        completion_result: AIMessage,
    ) -> ExecutionPlan:
        """解析 LLM planner 的结构化结果。"""

        payload = self._extract_json_payload(completion_result.content)
        requested_tool_names = state.get("requested_tool_names") or []
  
        primary_category = self._coerce_primary_category(payload.get("primary_category"))
        if primary_category is None:
            primary_category = "general"

        # 构建默认步骤
        fallback_steps = self._build_steps(
            primary_category=primary_category,
            has_requested_tools=bool(requested_tool_names)
        )
        # LLM 输出的步骤可能不完整，使用默认步骤填充。
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
        # 规范化大模型生成内容中提取 JSON 中的 step
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

        # 防止步骤中缺少answer步骤，无法路由到最终的终结节点
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
        if primary_category == "service_area":
            return "service"
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
            if step.executor in {"route", "traffic", "service", "report", "tool", "mcp"}:
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
        has_requested_tools: bool
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
                    goal="检索通过知识库检索相关问题知识",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结知识库的检索结果并回答用户",
                    depends_on=["rag_1"],
                ),
            ]

        if primary_category == "route_planning":
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

        if primary_category == "service_area":
            return [
                ExecutionStep(
                    step_id="service_1",
                    executor="service",
                    goal="查询服务区、充电和商业配套信息",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结服务区结果并回答用户",
                    depends_on=["service_1"],
                ),
            ]

        if primary_category == "network_report":
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

    def _build_fallback_plan(self, state: AgentState) -> ExecutionPlan:
        """在 LLM planner 不可用时，使用规则生成保底计划。"""

        requested_tool_names = state.get("requested_tool_names") or []
        primary_category = self._infer_primary_category(
            latest_user_message=str(state.get("latest_user_message", "")),
            has_requested_tools=bool(requested_tool_names),
        )
        steps = self._build_steps(
            primary_category=primary_category,
            has_requested_tools=bool(requested_tool_names),
        )
        return ExecutionPlan(
            primary_category=primary_category,
            execution_mode=self._resolve_execution_mode(steps),
            recommended_route=self._derive_recommended_route(
                steps=steps,
                primary_category=primary_category,
                has_requested_tools=bool(requested_tool_names),
            ),
            steps=steps,
        )

    @staticmethod
    def _infer_primary_category(
        *,
        latest_user_message: str,
        has_requested_tools: bool,
    ) -> ProblemCategory:
        """根据问题文本做最小规则分类。"""

        if has_requested_tools:
            return "general"

        normalized_message = latest_user_message.strip().lower()
        if (
            latest_user_message.startswith("知识库:")
            or normalized_message.startswith("knowledge:")
            or normalized_message.startswith("konwledge:")
            or any(keyword in latest_user_message for keyword in ("政策", "制度", "标准", "规范", "口径"))
        ):
            return "policy"
        if any(
            keyword in latest_user_message
            for keyword in ("服务区", "充电桩", "充电站", "休息区", "加油站", "便利店")
        ):
            return "service_area"
        if any(
            keyword in latest_user_message
            for keyword in ("全路网", "路网", "日报", "周报", "月报", "表格", "对比")
        ):
            return "network_report"
        if any(
            keyword in latest_user_message
            for keyword in ("路况", "拥堵", "封闭", "施工", "事故", "缓行", "通行情况")
        ):
            return "traffic_status"
        if (
            "到" in latest_user_message
            and any(keyword in latest_user_message for keyword in ("怎么走", "怎么去", "路线", "导航"))
        ):
            return "route_planning"
        return "general"

