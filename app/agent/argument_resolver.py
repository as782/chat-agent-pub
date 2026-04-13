"""Agent 参数提取模块。

负责根据 planner 给出的业务分类，对当前问题做最小结构化参数提取。
当前阶段先使用规则方式输出稳定骨架，后续可替换为 LLM argument resolver。
"""

from __future__ import annotations

import re

from app.agent.state import AgentState, ExecutorType, ProblemCategory, ResolvedArguments
from app.clients.llm_client import LlmInputMessage

ROUTE_PAIR_PATTERN = re.compile(
    r"(?:从)?(?P<origin>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})"
    r"(?P<connector>到|至|前往|去往|往|去)"
    r"(?P<destination>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})"
)
_ROUTE_NON_PLACE_KEYWORDS = (
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
        match = self._extract_od_pair(normalized_query)

        arguments: dict[str, object] = {"query": normalized_query}
        missing_fields: list[str] = []

        if match is not None:
            arguments["origin"] = match["origin"]
            arguments["destination"] = match["destination"]
        else:
            missing_fields.extend(["origin", "destination"])

        arguments["travel_mode"] = self._infer_travel_mode(normalized_query)

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
            "road": self._normalize_traffic_target(normalized_query),
            "target": self._normalize_traffic_target(normalized_query),
        }
        time_range = self._infer_time_range(normalized_query)
        if time_range is not None:
            arguments["time_range"] = time_range
        return ResolvedArguments(category=category, arguments=arguments)

    def _resolve_service_arguments(
        self,
        latest_user_message: str,
        *,
        category: ProblemCategory,
    ) -> ResolvedArguments:
        """提取服务区问题中的服务区或设施关键词。"""

        normalized_query = self._strip_prefix(latest_user_message, prefixes=("service:",))
        keyword = self._infer_service_keyword(normalized_query)
        if keyword is None:
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
        arguments: dict[str, object] = {"query": normalized_query}

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

    @classmethod
    def _extract_od_pair(cls, message: str) -> dict[str, str] | None:
        """抽取路线 OD 信息，避免把时间短语误识别为起终点。"""

        normalized_message = message.strip()
        match = ROUTE_PAIR_PATTERN.search(normalized_message)
        if match is None:
            return None

        origin = cls._clean_route_place(match.group("origin"))
        destination = cls._clean_route_place(match.group("destination"))
        if not origin or not destination:
            return None
        if cls._contains_non_place_context(origin) or cls._contains_non_place_context(destination):
            return None
        if origin == destination:
            return None
        return {"origin": origin, "destination": destination}

    @staticmethod
    def _contains_non_place_context(value: str) -> bool:
        """过滤明显不是地名的片段。"""

        return any(keyword in value for keyword in _ROUTE_NON_PLACE_KEYWORDS)

    @staticmethod
    def _infer_travel_mode(message: str) -> str:
        """根据问题文本推断出行方式。"""

        if any(keyword in message for keyword in ("公交", "地铁", "轻轨")):
            return "public_transit"
        if any(keyword in message for keyword in ("步行",)):
            return "walking"
        if any(keyword in message for keyword in ("骑行", "骑车", "自行车")):
            return "cycling"
        if any(keyword in message for keyword in ("开车", "驾车", "自驾", "汽车")):
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
    def _normalize_traffic_target(message: str) -> str:
        """清理路况查询目标，保留更接近道路或路线的文本。"""

        target = message.strip()
        for token in ("今天", "明天", "昨天", "当前", "现在", "实时", "目前", "此刻"):
            target = target.replace(token, " ")
        for token in (
            "大客车",
            "中客车",
            "小客车",
            "货车",
            "客车",
            "轿车",
            "新能源车",
            "摩托车",
            "非机动车",
        ):
            index = target.find(token)
            if index > 0:
                target = target[:index].strip()
                break
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
            "限行了吗",
            "限行么",
            "限行吗",
            "限行没",
            "限行",
            "禁行",
            "管制",
            "封闭",
        ):
            if target.endswith(suffix):
                target = target[: -len(suffix)].strip()
        return " ".join(target.split()).strip()

    @staticmethod
    def _infer_service_keyword(message: str) -> str | None:
        """推断服务区查询关键词。"""

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
    def _extract_reference_answer(input_messages: list[LlmInputMessage]) -> str | None:
        """从显式输入消息中提取最近一条 assistant 内容作为参考答案。"""

        for message in reversed(input_messages):
            if message.role != "assistant":
                continue
            content = message.content.strip()
            if content:
                return content
        return None
