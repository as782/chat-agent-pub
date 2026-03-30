"""Agent 参数提取模块。

负责根据 planner 给出的业务分类，对当前问题做最小结构化参数提取。
当前阶段先使用规则方式输出稳定骨架，后续可替换为 LLM argument resolver。
"""

from __future__ import annotations

import re

from app.agent.state import AgentState, ExecutorType, ProblemCategory, ResolvedArguments
from app.clients.llm_client import LlmInputMessage

ROUTE_PAIR_PATTERN = re.compile(
    r"(?:从)?(?P<origin>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})到(?P<destination>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})"
)


class ArgumentResolver:
    """规则式参数提取器。"""

    def resolve(self, state: AgentState) -> ResolvedArguments:
        """根据当前主分类提取结构化参数。"""

        category = state.get("primary_category", "general")
        latest_user_message = str(state.get("latest_user_message", ""))

        if category == "route_planning":
            return self._resolve_route_arguments(latest_user_message, category=category)
        if category == "traffic_status":
            return self._resolve_traffic_arguments(latest_user_message, category=category)
        if category == "service_area":
            return self._resolve_service_arguments(latest_user_message, category=category)
        if category == "network_report":
            return self._resolve_report_arguments(latest_user_message, category=category)
        if category == "policy":
            return self._resolve_policy_arguments(latest_user_message, category=category)
        return ResolvedArguments(category=category, arguments={"query": latest_user_message})

    def resolve_for_executor(
        self,
        state: AgentState,
        *,
        executor: ExecutorType,
    ) -> ResolvedArguments:
        """根据执行器类型提取对应的结构化参数。"""

        latest_user_message = str(state.get("latest_user_message", ""))
        input_messages = state.get("input_messages", [])

        if executor == "rag":
            return self._resolve_policy_arguments(latest_user_message, category="policy")
        if executor in {"mcp", "route"}:
            return self._resolve_route_arguments(
                latest_user_message,
                category="route_planning",
            )
        if executor == "traffic":
            return self._resolve_traffic_arguments(
                latest_user_message,
                category="traffic_status",
            )
        if executor == "service":
            return self._resolve_service_arguments(
                latest_user_message,
                category="service_area",
            )
        if executor == "report":
            return self._resolve_report_arguments(
                latest_user_message,
                category="network_report",
                input_messages=input_messages if isinstance(input_messages, list) else [],
            )
        return ResolvedArguments(category="general", arguments={"query": latest_user_message})

    def _resolve_route_arguments(
        self,
        latest_user_message: str,
        *,
        category: ProblemCategory,
    ) -> ResolvedArguments:
        """提取路线规划问题中的起终点和出行方式。"""

        normalized_query = self._strip_prefix(latest_user_message, prefixes=("mcp:",))
        match = ROUTE_PAIR_PATTERN.search(normalized_query)

        arguments: dict[str, object] = {"query": normalized_query}
        missing_fields: list[str] = []

        if match:
            arguments["origin"] = self._clean_route_place(match.group("origin"))
            arguments["destination"] = self._clean_route_place(match.group("destination"))
        else:
            missing_fields.extend(["origin", "destination"])

        if "公交" in normalized_query or "地铁" in normalized_query:
            arguments["travel_mode"] = "public_transit"
        elif "驾车" in normalized_query or "开车" in normalized_query:
            arguments["travel_mode"] = "driving"
        elif "步行" in normalized_query:
            arguments["travel_mode"] = "walking"
        else:
            arguments["travel_mode"] = "auto"

        return ResolvedArguments(
            category=category,
            arguments=arguments,
            missing_fields=missing_fields,
        )

    def _resolve_traffic_arguments(
        self,
        latest_user_message: str,
        *,
        category: ProblemCategory,
    ) -> ResolvedArguments:
        """提取路况问题中的查询对象和时间范围。"""

        normalized_query = self._strip_prefix(latest_user_message, prefixes=("traffic:",))
        arguments: dict[str, object] = {
            "query": normalized_query,
            "road": normalized_query.replace("路况", "").replace("怎么样", "").replace("如何", "").strip(),
            "target": normalized_query.replace("路况", "").replace("怎么样", "").strip(),
        }
        if "实时" in normalized_query or "当前" in normalized_query:
            arguments["time_range"] = "current"
        if "今天" in normalized_query:
            arguments["time_range"] = "today"
        return ResolvedArguments(category=category, arguments=arguments)

    def _resolve_service_arguments(
        self,
        latest_user_message: str,
        *,
        category: ProblemCategory,
    ) -> ResolvedArguments:
        """提取服务区问题中的服务区或设施关键词。"""

        normalized_query = self._strip_prefix(latest_user_message, prefixes=("service:",))
        keyword = normalized_query
        for suffix in ("服务区", "充电桩", "充电站", "有什么", "怎么样", "信息", "情况"):
            keyword = keyword.replace(suffix, " ")
        keyword = " ".join(keyword.split()).strip()
        if not keyword:
            keyword = normalized_query.strip()
        return ResolvedArguments(
            category=category,
            arguments={"query": normalized_query, "keyword": keyword},
        )

    def _resolve_report_arguments(
        self,
        latest_user_message: str,
        *,
        category: ProblemCategory,
        input_messages: list[LlmInputMessage] | None = None,
    ) -> ResolvedArguments:
        """提取路网报告问题中的报表参数。"""

        normalized_query = latest_user_message.strip()
        arguments: dict[str, object] = {
            "query": normalized_query,
            "scope": "全路网" if "全路网" in normalized_query else "未明确范围",
            "need_table": "表格" in normalized_query or "表" in normalized_query,
            "need_comparison": "对比" in normalized_query or "上次" in normalized_query,
        }
        if "日报" in normalized_query:
            arguments["report_type"] = "daily"
        elif "周报" in normalized_query:
            arguments["report_type"] = "weekly"
        elif "月报" in normalized_query:
            arguments["report_type"] = "monthly"
        else:
            arguments["report_type"] = "ad_hoc"

        reference_answer = self._extract_reference_answer(input_messages or [])
        if reference_answer is not None:
            arguments["reference_answer"] = reference_answer

        return ResolvedArguments(category=category, arguments=arguments)

    def _resolve_policy_arguments(
        self,
        latest_user_message: str,
        *,
        category: ProblemCategory,
    ) -> ResolvedArguments:
        """提取政策问答中的检索查询文本。"""

        normalized_query = self._strip_prefix(
            latest_user_message,
            prefixes=("知识库:", "knowledge:", "konwledge:"),
        )
        return ResolvedArguments(category=category, arguments={"query": normalized_query})

    @staticmethod
    def _strip_prefix(message: str, *, prefixes: tuple[str, ...]) -> str:
        """去掉显式前缀，保留真正业务查询文本。"""

        stripped_message = message.strip()
        lowered_message = stripped_message.lower()
        for prefix in prefixes:
            if lowered_message.startswith(prefix):
                return stripped_message[len(prefix) :].strip()
        return stripped_message

    @staticmethod
    def _clean_route_place(value: str) -> str:
        """清理路线提取结果中的尾部语气词。"""

        cleaned_value = value.strip()
        for suffix in ("怎么走", "怎么去", "如何走", "如何去", "路线", "路况"):
            if cleaned_value.endswith(suffix):
                cleaned_value = cleaned_value[: -len(suffix)].strip()
        return cleaned_value

    @staticmethod
    def _extract_reference_answer(input_messages: list[LlmInputMessage]) -> str | None:
        """从显式输入消息中提取最近一条 assistant 内容作为参考答案。"""

        for message in reversed(input_messages):
            if message.role != "assistant":
                continue
            content = message.content.strip()
            if content:
                return content
        return None
