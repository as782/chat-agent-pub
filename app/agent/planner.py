"""Agent 规划器模块。
负责根据用户问题给出业务分类和最小执行计划。
当前阶段先提供规则式规划器骨架，为后续接入 LLM planner 预留统一输出结构。
"""

from __future__ import annotations

from app.agent.state import AgentRoute, AgentState, ExecutionPlan, ExecutionStep, ProblemCategory

POLICY_KEYWORDS = ("政策", "标准", "规范", "制度", "规定", "依据", "口径")
ROUTE_KEYWORDS = ("怎么走", "路线", "导航", "到", "路径", "方案")
TRAFFIC_KEYWORDS = ("路况", "拥堵", "封闭", "施工", "事故", "缓行", "通行")
NETWORK_REPORT_KEYWORDS = ("全路网", "日报", "周报", "月报", "汇总", "表格", "对比", "分析")


class PlannerService:
    """规则式规划器。

    当前先使用显式规则给出分类和执行计划，后续可以替换为 LLM planner，
    但仍然复用同一套 ExecutionPlan 输出结构。
    """

    def build_plan(self, state: AgentState) -> ExecutionPlan:
        """根据当前状态生成分类与执行计划。"""

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
        )
        execution_mode = "direct" if len(steps) <= 1 else "single_step"
        if len(steps) > 2:
            execution_mode = "multi_step"

        return ExecutionPlan(
            primary_category=primary_category,
            execution_mode=execution_mode,
            recommended_route=recommended_route,
            steps=steps,
        )

    def _detect_primary_category(
        self,
        *,
        latest_user_message: str,
        normalized_message: str,
        has_requested_tools: bool,
    ) -> ProblemCategory:
        """识别当前问题的主业务分类。"""

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
        if any(keyword in latest_user_message for keyword in TRAFFIC_KEYWORDS):
            return "traffic_status"
        if any(keyword in latest_user_message for keyword in POLICY_KEYWORDS):
            return "policy"
        if any(keyword in latest_user_message for keyword in ROUTE_KEYWORDS):
            return "route_planning"
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
            return "mcp"
        if primary_category == "traffic_status":
            return "traffic"
        if primary_category == "network_report":
            return "report"
        return "answer"

    @staticmethod
    def _build_steps(
        *,
        primary_category: ProblemCategory,
        has_requested_tools: bool,
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
            return [
                ExecutionStep(
                    step_id="mcp_1",
                    executor="mcp",
                    goal="查询路线规划相关数据",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结路线结果并回答用户",
                    depends_on=["mcp_1"],
                ),
            ]

        if primary_category == "traffic_status":
            return [
                ExecutionStep(
                    step_id="traffic_1",
                    executor="tool",
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
