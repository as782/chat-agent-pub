"""Agent 规划器模块。
负责根据用户问题给出业务分类和最小执行计划。
当前同时支持规则式规划和可选的 LLM planner，并统一输出 ExecutionPlan。
"""

from __future__ import annotations

from json import JSONDecodeError, dumps, loads
from re import DOTALL, compile as re_compile, search
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
_NETWORK_SCOPE_KEYWORDS: tuple[str, ...] = (
    "全路网",
    "路网",
    "全省",
    "省内",
    "整体",
    "总体",
    "总体情况",
    "全省内",
    "全部线路",
    "所有线路",
    "全域",
    "全网",
    "概况",
    "汇总",
    "汇总情况",
)
_NETWORK_TRAFFIC_KEYWORDS: tuple[str, ...] = (
    "路况",
    "高速",
    "通行",
    "拥堵",
    "事故",
    "施工",
    "封闭",
    "缓行",
    "实时",
)
_TIME_QUERY_KEYWORDS: tuple[str, ...] = (
    "当前时间",
    "现在时间",
    "现在几点",
    "几点了",
    "多少点",
    "现在多少点",
    "日期",
    "今天几号",
    "今天日期",
    "现在日期",
    "当前日期",
    "时间",
)
_CALCULATION_KEYWORDS: tuple[str, ...] = (
    "计算",
    "算一下",
    "帮我算",
)
_OD_ROUTE_PATTERNS = (
    re_compile(
        r"(?:从)?(?P<origin>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})"
        r"(?P<connector>到|至|前往|去往|往|去|回)"
        r"(?P<destination>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})"
    ),
)
_ROUTE_INTENT_KEYWORDS: tuple[str, ...] = (
    "怎么走",
    "怎么去",
    "如何走",
    "如何去",
    "路线",
    "导航",
    "通行",
    "堵不堵",
    "堵吗",
    "拥堵吗",
    "堵",
)
_NON_ROUTE_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "怎么",
    "如何",
    "哪",
    "哪里",
    "去哪里",
    "从哪",
    "从哪里",
    "今天",
    "明天",
    "昨天",
    "昨日",
    "现在",
    "当前",
    "目前",
    "早上",
    "上午",
    "中午",
    "下午",
    "晚上",
    "凌晨",
    "节假日",
    "收费",
    "过路费",
    "通行费",
    "免费",
)


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
            plan = await self._build_plan_with_llm(state)
        except Exception as exception:  # noqa: BLE001
            LOGGER.warning(
                "LLM planner 规划失败, error=%s",
                str(exception)
            )
            plan = self._build_fallback_plan(state)

        LOGGER.info(
            "Planner final execution plan: %s",
            dumps(self._serialize_execution_plan(plan), ensure_ascii=False),
        )
        return plan

    async def _build_plan_with_llm(self, state: AgentState) -> ExecutionPlan:
        """调用 LLM 生成规划结果。"""

        if self._llm_client is None:
            raise RuntimeError("LLM planner 未注入 llm_client。")

        planner_api_key = self._settings.planner_api_key
        planner_timeout_seconds = self._settings.planner_timeout_seconds
        completion_result = await self._llm_client.create_chat_completion(
            messages=self._build_planner_messages(state),
            model_name=self._settings.planner_model,
            base_url=self._settings.planner_base_url or self._settings.openai_base_url,
            api_key=(
                planner_api_key.get_secret_value().strip() or None
                if planner_api_key is not None
                else None
            ),
            timeout_seconds=(
                planner_timeout_seconds
                if planner_timeout_seconds is not None
                else self._settings.openai_timeout_seconds
            ),
            enable_thinking=(
                self._settings.planner_enable_thinking
                if self._settings.planner_enable_thinking is not None
                else False
            ),
        )
        LOGGER.info(
            "Planner LLM response received: content=%s reasoning_content=%s",
            self._extract_message_text(completion_result),
            self._extract_reasoning_text(completion_result) or "",
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

    @staticmethod
    def _extract_message_text(message: AIMessage) -> str:
        """稳定提取 planner LLM 的文本内容用于日志打印。"""

        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            text_parts: list[str] = []
            for part in message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return str(message.content)

    @staticmethod
    def _extract_reasoning_text(message: AIMessage) -> str | None:
        """提取 planner LLM 的 reasoning_content，便于联调排查。"""

        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
        reasoning_content = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content or None
        return None

    def _parse_llm_plan(
        self,
        state: AgentState,
        completion_result: AIMessage,
    ) -> ExecutionPlan:
        """解析 LLM planner 的结构化结果。"""

        payload = self._extract_json_payload(completion_result.content)
        requested_tool_names = state.get("requested_tool_names") or []
        latest_user_message = str(state.get("latest_user_message", ""))

        primary_category = self._coerce_primary_category(payload.get("primary_category"))
        if primary_category is None:
            primary_category = "general"
        general_tool_name = (
            self._resolve_general_tool_name(latest_user_message)
            if primary_category == "general" and not requested_tool_names
            else None
        )

        steps = self._coerce_steps(
            payload.get("steps"),
            latest_user_message=latest_user_message,
            primary_category=primary_category,
        )
        if not steps:
            steps = self._build_steps(
                primary_category=primary_category,
                has_requested_tools=bool(requested_tool_names),
                general_tool_name=general_tool_name,
                latest_user_message=latest_user_message,
            )
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
    def _serialize_execution_plan(plan: ExecutionPlan) -> dict[str, object]:
        """Convert execution plan into a log-friendly JSON payload."""

        return {
            "primary_category": plan.primary_category,
            "execution_mode": plan.execution_mode,
            "recommended_route": plan.recommended_route,
            "need_clarification": plan.need_clarification,
            "clarification_question": plan.clarification_question,
            "steps": [
                {
                    "step_id": step.step_id,
                    "executor": step.executor,
                    "goal": step.goal,
                    "depends_on": list(step.depends_on),
                    "can_run_in_parallel": step.can_run_in_parallel,
                    "metadata": dict(step.metadata),
                }
                for step in plan.steps
            ],
        }

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
        latest_user_message: str = "",
        primary_category: ProblemCategory = "general",
    ) -> list[ExecutionStep]:
        """把 LLM 输出的 steps 规整为统一步骤对象。"""

        if not isinstance(value, list):
            return []

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
            normalized_metadata = self._enrich_step_metadata(
                executor=executor,
                metadata=normalized_metadata,
                latest_user_message=latest_user_message,
                primary_category=primary_category,
            )

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
            return []

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

    def _build_steps(
        self,
        *,
        primary_category: ProblemCategory,
        has_requested_tools: bool,
        general_tool_name: str | None = None,
        latest_user_message: str = "",
    ) -> list[ExecutionStep]:
        """根据主分类生成最小可执行步骤。"""

        if has_requested_tools:
            return [
                ExecutionStep(
                    step_id="tool_1",
                    executor="tool",
                    goal="执行用户显式开放的工具",
                    metadata=self._enrich_step_metadata(
                        executor="tool",
                        metadata={},
                        latest_user_message=latest_user_message,
                        primary_category=primary_category,
                    ),
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="根据工具结果生成最终回答",
                    depends_on=["tool_1"],
                ),
            ]

        if primary_category == "general" and general_tool_name is not None:
            return [
                ExecutionStep(
                    step_id="tool_1",
                    executor="tool",
                    goal=f"调用内置工具 {general_tool_name} 获取结果",
                    metadata=self._enrich_step_metadata(
                        executor="tool",
                        metadata={"preferred_tool": general_tool_name},
                        latest_user_message=latest_user_message,
                        primary_category=primary_category,
                    ),
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="根据工具结果生成最终回答",
                    depends_on=["tool_1"],
                ),
            ]

        if primary_category == "policy":
            if PlannerService._looks_like_od_query(latest_user_message):
                return [
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="查询起点到终点的推荐路线",
                        metadata=self._enrich_step_metadata(
                            executor="route",
                            metadata={},
                            latest_user_message=latest_user_message,
                            primary_category=primary_category,
                        ),
                    ),
                    ExecutionStep(
                        step_id="rag_1",
                        executor="rag",
                        goal="检索相关政策、收费或通行规则",
                        metadata=self._enrich_step_metadata(
                            executor="rag",
                            metadata={},
                            latest_user_message=latest_user_message,
                            primary_category=primary_category,
                        ),
                    ),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="综合路线和政策结果回答用户",
                        depends_on=["route_1", "rag_1"],
                    ),
                ]
            return [
                ExecutionStep(
                    step_id="rag_1",
                    executor="rag",
                    goal="检索通过知识库检索相关问题知识",
                    metadata=self._enrich_step_metadata(
                        executor="rag",
                        metadata={},
                        latest_user_message=latest_user_message,
                        primary_category=primary_category,
                    ),
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
                    metadata=self._enrich_step_metadata(
                        executor="route",
                        metadata={},
                        latest_user_message=latest_user_message,
                        primary_category=primary_category,
                    ),
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结路线结果并回答用户",
                    depends_on=["route_1"],
                ),
            ]

        if primary_category == "traffic_status":
            if PlannerService._looks_like_od_query(latest_user_message):
                return [
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="规划起点到终点的路线并提取沿途道路",
                        metadata=self._enrich_step_metadata(
                            executor="route",
                            metadata={},
                            latest_user_message=latest_user_message,
                            primary_category=primary_category,
                        ),
                    ),
                    ExecutionStep(
                        step_id="traffic_1",
                        executor="traffic",
                        goal="根据路线查询主线路况和拥堵信息",
                        depends_on=["route_1"],
                        metadata=self._enrich_step_metadata(
                            executor="traffic",
                            metadata={},
                            latest_user_message=latest_user_message,
                            primary_category=primary_category,
                        ),
                    ),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="结合路线和路况结果回答用户",
                        depends_on=["traffic_1"],
                    ),
                ]
            return [
                ExecutionStep(
                    step_id="traffic_1",
                    executor="traffic",
                    goal="查询路况或实时交通数据",
                    metadata=self._enrich_step_metadata(
                        executor="traffic",
                        metadata={},
                        latest_user_message=latest_user_message,
                        primary_category=primary_category,
                    ),
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结路况结果并回答用户",
                    depends_on=["traffic_1"],
                ),
            ]

        if primary_category == "service_area":
            if PlannerService._looks_like_od_query(latest_user_message):
                return [
                    ExecutionStep(
                        step_id="route_1",
                        executor="route",
                        goal="规划起点到终点的路线并提取沿途服务区",
                        metadata=self._enrich_step_metadata(
                            executor="route",
                            metadata={},
                            latest_user_message=latest_user_message,
                            primary_category=primary_category,
                        ),
                    ),
                    ExecutionStep(
                        step_id="service_1",
                        executor="service",
                        goal="根据路线查询沿途服务区、充电和配套信息",
                        depends_on=["route_1"],
                        metadata=self._enrich_step_metadata(
                            executor="service",
                            metadata={},
                            latest_user_message=latest_user_message,
                            primary_category=primary_category,
                        ),
                    ),
                    ExecutionStep(
                        step_id="answer_1",
                        executor="answer",
                        goal="综合路线和服务区结果回答用户",
                        depends_on=["service_1"],
                    ),
                ]
            return [
                ExecutionStep(
                    step_id="service_1",
                    executor="service",
                    goal="查询服务区、充电和商业配套信息",
                    metadata=self._enrich_step_metadata(
                        executor="service",
                        metadata={},
                        latest_user_message=latest_user_message,
                        primary_category=primary_category,
                    ),
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
                    metadata=self._enrich_step_metadata(
                        executor="report",
                        metadata={},
                        latest_user_message=latest_user_message,
                        primary_category=primary_category,
                    ),
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
                metadata=self._enrich_step_metadata(
                    executor="answer",
                    metadata={},
                    latest_user_message=latest_user_message,
                    primary_category=primary_category,
                ),
            )
        ]

    def _enrich_step_metadata(
        self,
        *,
        executor: ExecutorType,
        metadata: dict[str, object],
        latest_user_message: str,
        primary_category: ProblemCategory,
    ) -> dict[str, object]:
        """为步骤补齐该 executor 需要的稳定参数。"""

        merged_metadata = dict(metadata)
        inferred_metadata = self._infer_step_metadata(
            executor=executor,
            latest_user_message=latest_user_message,
            primary_category=primary_category,
        )
        for key, value in inferred_metadata.items():
            if key not in merged_metadata or self._is_empty_metadata_value(merged_metadata[key]):
                merged_metadata[key] = value
        return merged_metadata

    @staticmethod
    def _infer_step_metadata(
        *,
        executor: ExecutorType,
        latest_user_message: str,
        primary_category: ProblemCategory,
    ) -> dict[str, object]:
        """根据 executor 和当前问题推导可执行的 metadata。"""

        normalized_message = latest_user_message.strip()

        if executor == "route":
            metadata: dict[str, object] = {
                "query": normalized_message,
                "query_intent": "route_planning",
            }
            od_pair = PlannerService._extract_od_pair(normalized_message)
            if od_pair is not None:
                metadata.update(od_pair)
            travel_mode = PlannerService._infer_travel_mode(normalized_message)
            if travel_mode is not None:
                metadata["travel_mode"] = travel_mode
            return metadata

        if executor == "traffic":
            metadata = {
                "query": normalized_message,
                "query_intent": "route_based_traffic"
                if PlannerService._looks_like_od_query(normalized_message)
                else "traffic_status",
            }
            target = PlannerService._normalize_traffic_target(normalized_message)
            if target:
                metadata["road"] = target
                metadata["target"] = target
            time_range = PlannerService._infer_time_range(normalized_message)
            if time_range is not None:
                metadata["time_range"] = time_range
            focus = PlannerService._infer_traffic_focus(normalized_message)
            if focus is not None:
                metadata["focus"] = focus
            return metadata

        if executor == "service":
            metadata = {
                "query": normalized_message,
                "query_intent": "service_lookup",
            }
            keyword = PlannerService._infer_service_keyword(normalized_message)
            if keyword is not None:
                metadata["keyword"] = keyword
            facility_type = PlannerService._infer_service_facility_type(normalized_message)
            if facility_type is not None:
                metadata["facility_type"] = facility_type
            return metadata

        if executor == "rag":
            query = PlannerService._strip_query_prefix(normalized_message)
            metadata = {
                "query": query,
                "query_type": PlannerService._infer_policy_query_type(normalized_message),
            }
            keywords = PlannerService._infer_policy_keywords(normalized_message)
            if keywords:
                metadata["keywords"] = keywords
            focus = PlannerService._infer_policy_focus(normalized_message)
            if focus is not None:
                metadata["focus"] = focus
            return metadata

        if executor == "report":
            metadata = {
                "query": normalized_message,
                "scope": PlannerService._infer_report_scope(normalized_message),
            }
            compare_mode = PlannerService._infer_report_compare_mode(normalized_message)
            if compare_mode is not None:
                metadata["compare_mode"] = compare_mode
            return metadata

        if executor in {"tool", "mcp"}:
            preferred_tool = PlannerService._resolve_general_tool_name(normalized_message)
            if preferred_tool is not None:
                return {"preferred_tool": preferred_tool}
            return {}

        if executor == "answer":
            focus = PlannerService._infer_answer_focus(normalized_message, primary_category)
            if focus is not None:
                return {"focus": focus}
            return {}

        return {}

    @staticmethod
    def _is_empty_metadata_value(value: object) -> bool:
        """判断 metadata 值是否为空。"""

        return value is None or value == "" or value == [] or value == {}

    @staticmethod
    def _strip_query_prefix(message: str) -> str:
        """移除常见知识库/前缀标记，保留纯查询文本。"""

        stripped_message = message.strip()
        lowered_message = stripped_message.lower()
        for prefix in ("知识库:", "knowledge:", "konwledge:"):
            if lowered_message.startswith(prefix):
                return stripped_message[len(prefix) :].strip()
        return stripped_message

    @staticmethod
    def _extract_od_pair(message: str) -> dict[str, str] | None:
        """抽取 OD 起终点。"""

        normalized_message = message.strip()
        for pattern in _OD_ROUTE_PATTERNS:
            match = pattern.search(normalized_message)
            if match is None:
                continue
            origin = PlannerService._clean_route_place(match.group("origin"))
            destination = PlannerService._clean_route_place(match.group("destination"))
            if not origin or not destination:
                continue
            if PlannerService._contains_non_route_context(origin) or PlannerService._contains_non_route_context(
                destination
            ):
                continue
            if origin == destination:
                continue
            return {"origin": origin, "destination": destination}
        return None

    @staticmethod
    def _clean_route_place(value: str) -> str:
        """清理路线提取结果中的尾部语气词。"""

        cleaned_value = value.strip()
        for suffix in (
            "前往",
            "去往",
            "往",
            "去",
            "怎么走",
            "怎么去",
            "如何走",
            "如何去",
            "路线",
            "路况",
            "堵不堵",
            "堵吗",
            "拥堵吗",
            "是否拥堵",
            "会不会堵",
            "通畅吗",
        ):
            if cleaned_value.endswith(suffix):
                cleaned_value = cleaned_value[: -len(suffix)].strip()
        return cleaned_value

    @staticmethod
    def _contains_non_route_context(value: str) -> bool:
        """排除明显不是地名的片段。"""

        return any(keyword in value for keyword in _NON_ROUTE_CONTEXT_KEYWORDS)

    @staticmethod
    def _infer_travel_mode(message: str) -> str | None:
        """根据问题文本推断出行方式。"""

        if any(keyword in message for keyword in ("公交", "地铁", "轻轨")):
            return "public_transit"
        if any(keyword in message for keyword in ("步行",)):
            return "walking"
        if any(keyword in message for keyword in ("骑行", "骑车", "自行车")):
            return "cycling"
        if any(keyword in message for keyword in ("开车", "驾车", "自驾", "汽车", "高速")):
            return "driving"
        return "auto"

    @staticmethod
    def _infer_time_range(message: str) -> str | None:
        """推断路况查询的时间范围。"""

        if any(keyword in message for keyword in ("当前", "现在", "实时", "此刻")):
            return "current"
        if "今天" in message:
            return "today"
        if "明天" in message:
            return "tomorrow"
        if "昨天" in message:
            return "yesterday"
        return None

    @staticmethod
    def _infer_traffic_focus(message: str) -> str | None:
        """推断路况问题的关注点。"""

        if any(keyword in message for keyword in ("堵", "拥堵", "缓行")):
            return "congestion"
        if any(keyword in message for keyword in ("事故",)):
            return "accident"
        if any(keyword in message for keyword in ("施工", "封闭", "管制")):
            return "control"
        if any(keyword in message for keyword in ("收费站",)):
            return "toll"
        return None

    @staticmethod
    def _normalize_traffic_target(message: str) -> str:
        """清理路况查询目标，保留更接近道路/路线的文本。"""

        target = message.strip()
        for token in ("今天", "明天", "昨天", "当前", "现在", "实时", "目前", "此刻"):
            target = target.replace(token, " ")
        for suffix in (
            "路况怎么样",
            "路况如何",
            "怎么样",
            "如何",
            "堵不堵",
            "堵吗",
            "拥堵吗",
            "通畅吗",
            "通不通畅",
            "是否拥堵",
            "会不会堵",
        ):
            if target.endswith(suffix):
                target = target[: -len(suffix)].strip()
        return " ".join(target.split()).strip()

    @staticmethod
    def _infer_service_keyword(message: str) -> str | None:
        """推断服务区查询的核心关键词。"""

        keyword = message.strip()
        for suffix in (
            "服务区",
            "充电桩",
            "充电站",
            "充电",
            "加油站",
            "餐厅",
            "商店",
            "便利店",
            "有什么",
            "有哪些",
            "怎么样",
            "情况",
            "信息",
        ):
            keyword = keyword.replace(suffix, " ")
        keyword = " ".join(keyword.split()).strip()
        return keyword or None

    @staticmethod
    def _infer_service_facility_type(message: str) -> str | None:
        """推断服务区问题关注的设施类型。"""

        if any(keyword in message for keyword in ("充电桩", "充电站", "充电")):
            return "charging"
        if any(keyword in message for keyword in ("加油站",)):
            return "fuel"
        if any(keyword in message for keyword in ("餐厅", "商店", "便利店", "超市")):
            return "commercial"
        return None

    @staticmethod
    def _infer_policy_query_type(message: str) -> str:
        """推断知识库检索类型。"""

        if any(keyword in message for keyword in ("收费", "过路费", "通行费", "免费", "节假日", "跨天", "收费规则")):
            return "policy_interpretation"
        if any(keyword in message for keyword in ("标准", "规范", "口径", "办法", "规定")):
            return "policy_interpretation"
        return "knowledge_query"

    @staticmethod
    def _infer_policy_keywords(message: str) -> list[str]:
        """从问题里提取政策检索关键词。"""

        keyword_candidates = (
            "高速过路费",
            "通行费",
            "过路费",
            "跨天",
            "收费规则",
            "免费时段",
            "节假日免费",
            "驶离出口时间",
            "清明节免费",
        )
        return [keyword for keyword in keyword_candidates if keyword in message]

    @staticmethod
    def _infer_policy_focus(message: str) -> str | None:
        """推断政策问题的回答焦点。"""

        if any(keyword in message for keyword in ("收费", "过路费", "通行费", "免费")):
            return "收费判断"
        if any(keyword in message for keyword in ("标准", "规范", "口径", "办法", "规定")):
            return "政策规则"
        return None

    @staticmethod
    def _infer_report_scope(message: str) -> str | None:
        """推断报表查询范围。"""

        if any(keyword in message for keyword in ("全路网", "路网", "全省", "全网", "整体", "汇总")):
            return "network"
        return None

    @staticmethod
    def _infer_report_compare_mode(message: str) -> str | None:
        """推断报表是否需要对比模式。"""

        if any(keyword in message for keyword in ("对比", "比较", "变化", "同比", "环比")):
            return "compare"
        return None

    @staticmethod
    def _infer_answer_focus(message: str, primary_category: ProblemCategory) -> str | None:
        """推断 answer 阶段的回答焦点。"""

        if any(keyword in message for keyword in ("收费", "过路费", "通行费", "免费")):
            return "收费判断"
        if any(keyword in message for keyword in ("堵", "拥堵", "缓行")):
            return "通行情况"
        if any(keyword in message for keyword in ("服务区", "充电桩", "充电站")):
            return "服务区设施"
        if primary_category == "network_report":
            return "路网汇总"
        if primary_category == "policy":
            return "政策规则"
        return None

    def _build_fallback_plan(self, state: AgentState) -> ExecutionPlan:
        """在 LLM planner 不可用时，使用规则生成保底计划。"""

        requested_tool_names = state.get("requested_tool_names") or []
        latest_user_message = str(state.get("latest_user_message", ""))
        primary_category = self._infer_primary_category(
            latest_user_message=latest_user_message,
            has_requested_tools=bool(requested_tool_names),
        )
        primary_category = self._normalize_primary_category(
            latest_user_message=latest_user_message,
            primary_category=primary_category,
        )
        general_tool_name = (
            self._resolve_general_tool_name(latest_user_message)
            if primary_category == "general" and not requested_tool_names
            else None
        )
        steps = self._build_steps(
            primary_category=primary_category,
            has_requested_tools=bool(requested_tool_names),
            general_tool_name=general_tool_name,
            latest_user_message=latest_user_message,
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
            for keyword in ("路况", "拥堵", "堵不堵", "堵吗", "堵", "封闭", "施工", "事故", "缓行", "通行情况", "通畅吗")
        ):
            return "traffic_status"
        if PlannerService._looks_like_od_query(latest_user_message):
            return "route_planning"
        if (
            "到" in latest_user_message
            and any(keyword in latest_user_message for keyword in ("怎么走", "怎么去", "路线", "导航"))
        ):
            return "route_planning"
        return "general"

    @staticmethod
    def _normalize_primary_category(
        *,
        latest_user_message: str,
        primary_category: ProblemCategory,
    ) -> ProblemCategory:
        """对全省/整体类路况问题做兜底纠偏。"""

        if PlannerService._should_force_network_report(latest_user_message):
            return "network_report"
        inferred_primary_category = PlannerService._infer_primary_category(
            latest_user_message=latest_user_message,
            has_requested_tools=False,
        )
        if primary_category in {"general", "route_planning"} and inferred_primary_category != "general":
            return inferred_primary_category
        return primary_category

    @staticmethod
    def _should_force_network_report(latest_user_message: str) -> bool:
        """识别更适合走全路网报表分支的问题。"""

        return any(keyword in latest_user_message for keyword in _NETWORK_SCOPE_KEYWORDS) and any(
            keyword in latest_user_message for keyword in _NETWORK_TRAFFIC_KEYWORDS
        )

    @staticmethod
    def _resolve_general_tool_name(latest_user_message: str) -> str | None:
        normalized_message = latest_user_message.strip().lower()
        normalized_expression = PlannerService._normalize_expression_text(normalized_message)
        if PlannerService._looks_like_time_query(latest_user_message, normalized_message):
            return "current_datetime"
        if PlannerService._looks_like_calculation_query(latest_user_message, normalized_expression):
            return "calculator"
        return None

    @staticmethod
    def _looks_like_time_query(latest_user_message: str, normalized_message: str) -> bool:
        return any(keyword in latest_user_message for keyword in _TIME_QUERY_KEYWORDS) or any(
            keyword in normalized_message for keyword in ("what time", "current time", "date")
        )

    @staticmethod
    def _looks_like_calculation_query(latest_user_message: str, normalized_message: str) -> bool:
        if any(keyword in latest_user_message for keyword in _CALCULATION_KEYWORDS):
            return True
        return search(r"^\s*[\d\.\(\)\+\-\*/%\s]+\s*$", normalized_message) is not None

    @staticmethod
    def _normalize_expression_text(message: str) -> str:
        """将常见中文算式符号规整成便于识别的表达式文本。"""

        normalized_message = (
            message.replace("（", "(")
            .replace("）", ")")
            .replace("＋", "+")
            .replace("－", "-")
            .replace("—", "-")
            .replace("–", "-")
            .replace("×", "*")
            .replace("x", "*")
            .replace("÷", "/")
            .replace("／", "/")
            .replace("％", "%")
        )
        while normalized_message and normalized_message[-1] in {"=", "＝", "?", "？", "。", "."}:
            normalized_message = normalized_message[:-1].rstrip()
        return normalized_message

    @staticmethod
    def _looks_like_od_query(latest_user_message: str) -> bool:
        """Detect origin-destination style queries that need route context."""

        return PlannerService._extract_od_pair(latest_user_message) is not None
