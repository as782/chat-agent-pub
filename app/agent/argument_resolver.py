"""Agent 参数提取模块。

负责根据 planner 给出的业务分类，对当前问题做最小结构化参数提取。
当前阶段先使用规则方式输出稳定骨架，后续可替换为 LLM argument resolver。
"""

from __future__ import annotations

import re

from app.agent.facility_catalog import load_default_facility_catalog
from app.agent.od import resolve_od
from app.agent.road_inference import infer_traffic_context
from app.agent.state import AgentState, ExecutorType, ProblemCategory, ResolvedArguments
from app.clients.llm_client import LlmInputMessage

ROUTE_PAIR_PATTERN = re.compile(
    r"(?:从)?(?P<origin>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})"
    r"(?P<connector>到|至|前往|去往|往|去)"
    r"(?P<destination>[\u4e00-\u9fffA-Za-z0-9·\-]{2,20})"
)
ROAD_SEGMENT_SPLIT_PATTERN = re.compile(r"(?:/|、|，|,|；|;|以及|及|和|与|跟|还是)")
ROAD_TOKEN_PATTERN = re.compile(
    r"(?:"
    r"[GS]\d{1,4}"
    r"|[\u4e00-\u9fff]{2,16}(?:高速公路|高速|绕城高速|绕城|环线高速|环线|快速路|国道|省道|大道|大桥|隧道|路段)"
    r")"
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
        od_resolution = resolve_od(normalized_query)
        match = (
            {"origin": od_resolution.origin, "destination": od_resolution.destination}
            if od_resolution.is_complete
            else self._extract_od_pair(normalized_query)
        )

        arguments: dict[str, object] = {
            "query": normalized_query,
            "od_resolution_source": od_resolution.source,
            "od_confidence": od_resolution.confidence,
        }
        missing_fields: list[str] = []

        if match is not None:
            arguments["origin"] = match["origin"]
            arguments["destination"] = match["destination"]
            if od_resolution.is_complete:
                arguments["origin_match_type"] = od_resolution.origin_match_type
                arguments["destination_match_type"] = od_resolution.destination_match_type
                if od_resolution.warnings:
                    arguments["od_warnings"] = list(od_resolution.warnings)
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
        normalized_target = self._normalize_traffic_target(normalized_query)
        roads = self._extract_traffic_roads(normalized_query)
        inferred_context = infer_traffic_context(
            message=normalized_query,
            normalized_target=normalized_target,
            explicit_roads=roads,
        )
        road = inferred_context.road
        if road is None and self._normalize_traffic_road_candidate(normalized_target) is not None:
            road = normalized_target
        arguments: dict[str, object] = {
            "query": normalized_query,
            "target": inferred_context.target or normalized_target,
        }
        if road is not None:
            arguments["road"] = road
        if inferred_context.roads:
            arguments["roads"] = list(inferred_context.roads)
        if inferred_context.direction is not None:
            arguments["direction"] = inferred_context.direction
        if inferred_context.toll_station is not None:
            arguments["toll_station"] = inferred_context.toll_station
            catalog = load_default_facility_catalog()
            toll_match = catalog.best_toll_station(
                f"{inferred_context.toll_station} {normalized_query}",
                source="resolver",
            )
            if toll_match is not None and not roads:
                arguments["toll_station"] = toll_match.canonical_name
                if toll_match.road_code:
                    arguments["road_code"] = toll_match.road_code
                    arguments["road"] = toll_match.road_code
                elif toll_match.road_name_core and not arguments.get("road"):
                    arguments["road"] = toll_match.road_name_core
                if toll_match.road_name_core:
                    arguments["road_name"] = toll_match.road_name_core
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
        catalog = load_default_facility_catalog()
        named_service_area = self._extract_named_service_area_candidate(normalized_query)
        catalog_lookup_query = named_service_area or normalized_query
        keyword = catalog.best_service_keyword(catalog_lookup_query, source="resolver")
        service_terms = catalog.resolve_service_query_terms(catalog_lookup_query)

        arguments: dict[str, object] = {"query": normalized_query}
        if named_service_area is not None:
            arguments["service_name"] = named_service_area
            arguments["catalog_service_match"] = keyword is not None

        if keyword is None and named_service_area is not None:
            keyword = named_service_area
        if keyword is None:
            keyword = self._infer_service_keyword(normalized_query)
        if keyword is None:
            keyword = normalized_query.strip()

        arguments["keyword"] = keyword
        if service_terms:
            arguments["service_query_terms"] = service_terms
        return ResolvedArguments(
            category=category,
            arguments=arguments,
        )

    @classmethod
    def _extract_named_service_area_candidate(cls, message: str) -> str | None:
        """提取用户明确指名的服务区/停车区名称，供目录强匹配使用。"""

        if cls._extract_od_pair(message) is not None:
            return None

        cleaned_message = message.strip()
        for token in (
            "有没有充电桩",
            "有没有快充",
            "有没有慢充",
            "充电桩多吗",
            "充电桩够吗",
            "充电方便吗",
            "充电情况",
            "充电状态",
            "充电站",
            "充电桩",
            "加油站",
            "商业配套",
            "配套服务",
            "配套",
            "有什么",
            "有哪些",
            "状态",
            "情况",
            "信息",
            "繁忙吗",
            "拥堵吗",
            "堵吗",
            "空吗",
            "多吗",
            "如何",
            "怎么样",
            "是否正常",
            "正常吗",
            "吗",
            "呢",
            "？",
            "?",
        ):
            cleaned_message = cleaned_message.replace(token, " ")
        cleaned_message = " ".join(cleaned_message.split()).strip()

        matches = re.findall(
            r"([A-Za-z0-9\u4e00-\u9fff·\-\(\)（）]{2,20}(?:服务区|停车区))",
            cleaned_message,
        )
        if not matches:
            return None

        candidate = max((match.strip() for match in matches if match.strip()), key=len, default="")
        if not candidate:
            return None
        if any(token in candidate for token in ("哪些", "什么", "附近", "沿线", "沿途")):
            return None
        if re.match(r"^[GS]\d{1,4}", candidate, re.IGNORECASE):
            return None
        return candidate

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
        for token in (
            "哪条路",
            "哪条",
            "哪个路段",
            "哪个",
            "那条路",
            "那条",
            "车流量",
            "流量",
            "更大",
            "比较大",
            "更堵",
            "更挤",
            "更严重",
            "对比",
            "比较",
        ):
            target = target.replace(token, " ")
        return " ".join(target.split()).strip()

    @staticmethod
    def _extract_traffic_roads(message: str) -> list[str]:
        """提取并列出现的多个道路/高速名称。"""

        normalized_target = ArgumentResolver._normalize_traffic_target(message)
        if not normalized_target:
            return []

        road_targets: list[str] = []
        for segment in ROAD_SEGMENT_SPLIT_PATTERN.split(normalized_target):
            segment = segment.strip()
            if not segment:
                continue
            matches = [
                road_target
                for road_target in (
                    ArgumentResolver._normalize_traffic_road_candidate(match.group(0))
                    for match in ROAD_TOKEN_PATTERN.finditer(segment)
                )
                if road_target is not None
            ]
            if matches:
                road_targets.append(max(matches, key=len))

        if len(road_targets) >= 2:
            return ArgumentResolver._deduplicate_strings(road_targets)

        matches = [
            road_target
            for road_target in (
                ArgumentResolver._normalize_traffic_road_candidate(match.group(0))
                for match in ROAD_TOKEN_PATTERN.finditer(normalized_target)
            )
            if road_target is not None
        ]
        return ArgumentResolver._deduplicate_strings(matches)

    @staticmethod
    def _normalize_traffic_road_candidate(value: str) -> str | None:
        """剔除请求性前缀和泛化高速词，避免把用户指令当成道路名。"""

        candidate = value.strip()
        if not candidate:
            return None

        for prefix in (
            "请提供",
            "请帮我看",
            "请帮我查",
            "帮我看",
            "帮我查",
            "帮忙看",
            "帮忙查",
            "麻烦看",
            "麻烦查",
            "请看",
            "请查",
            "查一下",
            "查下",
            "看一下",
            "看看",
            "给我看",
            "给我查",
        ):
            if candidate.startswith(prefix) and len(candidate) > len(prefix):
                candidate = candidate[len(prefix) :].strip()
                break

        if candidate in {"高速", "高速路", "高速公路", "路况", "实时路况"}:
            return None
        return candidate

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

    @staticmethod
    def _deduplicate_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered_values: list[str] = []
        for value in values:
            normalized_value = value.strip()
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            ordered_values.append(normalized_value)
        return ordered_values

    @staticmethod
    def _clean_route_place(value: str) -> str:
        cleaned_value = value.strip()
        for prefix in (
            "我从",
            "从",
            "我想从",
            "想从",
            "准备从",
            "打算从",
            "计划从",
            "我想",
            "想",
        ):
            if cleaned_value.startswith(prefix) and len(cleaned_value) > len(prefix):
                cleaned_value = cleaned_value[len(prefix) :].strip()
                break

        for prefix in ("回",):
            if cleaned_value.startswith(prefix) and len(cleaned_value) > len(prefix):
                cleaned_value = cleaned_value[len(prefix) :].strip()
                break

        for suffix in (
            "前往",
            "去往",
            "往",
            "去",
            "怎么走",
            "怎么去",
            "如何走",
            "如何去",
            "怎么开",
            "如何开",
            "路线",
            "路况",
            "堵车吗",
            "堵不堵",
            "堵吗",
            "拥堵吗",
            "会不会堵",
            "正常通行吗",
            "正常吗",
            "可以上吗",
            "是正常的吗",
            "能看一下吗",
            "看一下吗",
            "怎么样",
            "咋样",
            "好吗",
            "吗",
            "？",
            "?",
        ):
            if cleaned_value.endswith(suffix):
                cleaned_value = cleaned_value[: -len(suffix)].strip()
        return cleaned_value

    @classmethod
    def _extract_od_pair(cls, message: str) -> dict[str, str] | None:
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

    def _resolve_traffic_arguments(
        self,
        latest_user_message: str,
        *,
        category: ProblemCategory,
    ) -> ResolvedArguments:
        normalized_query = self._strip_prefix(latest_user_message, prefixes=("traffic:",))
        normalized_target = self._normalize_traffic_target(normalized_query)
        roads = self._extract_traffic_roads(normalized_query)
        inferred_context = infer_traffic_context(
            message=normalized_query,
            normalized_target=normalized_target,
            explicit_roads=roads,
        )
        query_intent = "route_based_traffic" if self._extract_od_pair(normalized_query) is not None else "traffic_status"
        arguments: dict[str, object] = {
            "query": normalized_query,
            "road": inferred_context.road or normalized_target,
            "target": inferred_context.target or normalized_target,
            "query_intent": query_intent,
        }
        if inferred_context.roads:
            arguments["roads"] = list(inferred_context.roads)
        if inferred_context.direction is not None:
            arguments["direction"] = inferred_context.direction
        if inferred_context.toll_station is not None:
            arguments["toll_station"] = inferred_context.toll_station
        time_range = self._infer_time_range(normalized_query)
        if time_range is not None:
            arguments["time_range"] = time_range
        return ResolvedArguments(category=category, arguments=arguments)

    @staticmethod
    def _normalize_traffic_target(message: str) -> str:
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
            "正常通行吗",
            "是正常的吗",
            "正常吗",
            "能看一下吗",
            "看一下吗",
            "怎么样",
            "咋样",
            "堵车吗",
            "堵不堵",
            "堵吗",
            "拥堵吗",
            "会不会堵",
            "畅通吗",
            "可以上吗",
            "能走吗",
            "好走吗",
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
        target = target.replace("那边", " ")
        return " ".join(target.split()).strip()

    @classmethod
    def _extract_od_pair(cls, message: str) -> dict[str, str] | None:
        normalized_message = message.strip()
        fallback_match = re.search(
            r"(?:^|[，,\s])(?:我从|从)?(?P<origin>[\u4e00-\u9fffA-Za-z0-9路\-]{2,20})"
            r"(?P<connector>到|至|前往|去往|往|去|回)"
            r"(?P<destination>[\u4e00-\u9fffA-Za-z0-9路\-]{2,20})",
            normalized_message,
        )
        match = fallback_match or ROUTE_PAIR_PATTERN.search(normalized_message)
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
    def _clean_route_place(value: str) -> str:
        cleaned_value = value.strip()
        for prefix in (
            "我从",
            "从",
            "我想从",
            "想从",
            "准备从",
            "打算从",
            "计划从",
            "我想",
            "想",
        ):
            if cleaned_value.startswith(prefix) and len(cleaned_value) > len(prefix):
                cleaned_value = cleaned_value[len(prefix) :].strip()
                break

        if cleaned_value.startswith("回") and len(cleaned_value) > 1:
            cleaned_value = cleaned_value[1:].strip()

        for suffix in (
            "怎么走不堵",
            "怎么走最快",
            "怎么走",
            "怎么去",
            "如何走",
            "如何去",
            "怎么开",
            "如何开",
            "走哪条路最快",
            "哪条路最快",
            "那条高速不堵",
            "哪条高速不堵",
            "那条高速",
            "哪条高速",
            "推荐一下路线",
            "推荐路线",
            "路线",
            "路况",
            "堵车吗",
            "堵不堵",
            "不堵",
            "堵吗",
            "拥堵吗",
            "会不会堵",
            "正常通行吗",
            "正常吗",
            "可以上吗",
            "是正常的吗",
            "能看一下吗",
            "看一下吗",
            "怎么样",
            "咋样",
            "最快",
            "好吗",
            "吗",
            "？",
            "?",
        ):
            if cleaned_value.endswith(suffix):
                cleaned_value = cleaned_value[: -len(suffix)].strip()
        return cleaned_value

    @staticmethod
    def _contains_non_place_context(value: str) -> bool:
        normalized_value = value.strip()
        return normalized_value in {
            "",
            "怎么",
            "如何",
            "哪里",
            "现在",
            "当前",
            "目前",
            "今天",
            "明天",
            "昨天",
            "几点",
            "收费",
            "免费",
        }
