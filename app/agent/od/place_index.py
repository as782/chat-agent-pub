"""基于 Aho-Corasick 的地点候选索引。

该模块只负责“在文本中找出可能是地点的片段”，不负责判断起点/终点。
OD 关系判断仍由 resolver 根据连接词和上下文完成。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import ahocorasick

from app.agent.facility_catalog import load_default_facility_catalog
from app.agent.od.models import PlaceCandidate
from app.agent.od.normalizer import clean_endpoint
from app.agent.od.patterns import CANDIDATE_PREFIX_PATTERN

DEFAULT_REGION_CATALOG_PATH = Path(__file__).parents[1] / "data" / "region_catalog.json"

# 未入词典但形态明确的高速设施，例如“上海收费站”“张三服务区”。
_GENERIC_FACILITY_PATTERN = re.compile(
    r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·\-（）()]{1,24}?"
    r"(?:服务区|停车区|收费站|收费口|主站|副站))"
)
_GENERIC_ROAD_PATTERN = re.compile(
    r"(?P<name>(?:[GS]\d{1,4})?[\u4e00-\u9fffA-Za-z0-9·\-（）()]{1,20}?"
    r"(?:高速|快速路|西线|东线|南线|北线))"
)
_LEADING_RELATION_PATTERN = re.compile(
    r"^(?:从|由|到|至|开到|开往|前往|去往|往|去|回|在|位于)+"
)
_RELATION_PATTERN = re.compile(r"(?:去往|前往|开往|开到|从|由|到|至|往|去|回|在|位于)")
_REGION_TRANSPORT_SUFFIXES = ("东", "南", "西", "北", "站", "虹桥")

_PLACE_TYPE_PRIORITY = {
    "service_area": 90,
    "toll_station": 90,
    "generic_facility": 70,
    "generic_transport_hub": 65,
    "generic_road": 65,
    "region": 60,
}
_FACILITY_TERM_TOKENS = ("服务区", "停车区", "收费站", "收费口", "主站", "副站")


class PlaceIndex:
    """地点候选索引。"""

    def __init__(self) -> None:
        self._automaton = ahocorasick.Automaton()
        self._indexed_terms: set[str] = set()
        self._build_from_facility_catalog()
        self._build_from_region_catalog()
        self._automaton.make_automaton()

    def find_candidates(self, query: str) -> list[PlaceCandidate]:
        """查找文本中的地点候选，并对重叠候选做最长优先去重。"""

        candidates = [
            self._candidate_from_match(end_index=end_index, payload=payload)
            for end_index, payload in self._automaton.iter(query)
        ]
        candidates.extend(self._find_generic_facilities(query))
        candidates.extend(self._find_region_transport_hubs(query, candidates))
        candidates.extend(self._find_generic_roads(query))
        return self._deduplicate_overlaps(candidates)

    def _build_from_facility_catalog(self) -> None:
        """把 facility_catalog 中的服务区/收费站别名加入 AC 自动机。"""

        payload = load_default_facility_catalog().to_json_payload()
        for record in payload.get("service_areas", []):
            if isinstance(record, dict):
                self._add_record_terms(record=record, place_type="service_area")
        for record in payload.get("toll_stations", []):
            if isinstance(record, dict):
                self._add_record_terms(record=record, place_type="toll_station")

    def _add_record_terms(self, *, record: dict[str, object], place_type: str) -> None:
        canonical_name = str(record.get("canonical_name") or "").strip()
        group_name = str(record.get("group_name") or "").strip()
        terms = [
            canonical_name,
            group_name,
            *self._string_items(record.get("aliases")),
            *self._string_items(record.get("preferred_query_terms")),
        ]
        for term in self._unique_terms(terms):
            if len(term) < 2 or not self._looks_like_facility_term(term):
                continue
            self._add_index_term(
                term,
                payload={
                    "term": term,
                    "canonical_name": canonical_name or term,
                    "place_type": place_type,
                    "priority": _PLACE_TYPE_PRIORITY[place_type],
                    "metadata": {
                        "group_name": group_name,
                        "road_code": record.get("road_code"),
                        "road_name": record.get("road_name_core"),
                    },
                },
            )

    def _build_from_region_catalog(self) -> None:
        """把行政区划名称加入 AC 自动机，补足城市/区县 OD 候选。"""

        if not DEFAULT_REGION_CATALOG_PATH.exists():
            return
        payload = json.loads(DEFAULT_REGION_CATALOG_PATH.read_text(encoding="utf-8"))
        regions = payload.get("regions", []) if isinstance(payload, dict) else []
        for region in regions:
            if not isinstance(region, dict):
                continue
            canonical_name = str(region.get("canonical_name") or "").strip()
            terms = [
                canonical_name,
                *self._string_items(region.get("aliases")),
            ]
            for term in self._unique_terms(terms):
                if len(term) < 2:
                    continue
                # region 优先级低于设施，避免“金华收费站”被拆成“金华”。
                self._add_index_term(
                    term,
                    payload={
                        "term": term,
                        "canonical_name": canonical_name or term,
                        "place_type": "region",
                        "priority": _PLACE_TYPE_PRIORITY["region"],
                        "metadata": {
                            "region_type": region.get("place_type"),
                            "parent": region.get("parent"),
                        },
                    },
                )

    def _add_index_term(self, term: str, *, payload: dict[str, object]) -> None:
        """同一 term 只入库一次。

        region_catalog 按 GB2260 顺序生成，省/市通常早于区县；
        这里保留更早出现的高层级候选，避免“黄山”被“黄山区”短名覆盖。
        """

        if term in self._indexed_terms:
            return
        self._indexed_terms.add(term)
        self._automaton.add_word(term, payload)

    @staticmethod
    def _looks_like_facility_term(term: str) -> bool:
        """只把明确设施形态的别名放入 AC，避免城市名误映射成设施。"""

        return any(token in term for token in _FACILITY_TERM_TOKENS)

    @staticmethod
    def _candidate_from_match(*, end_index: int, payload: dict[str, object]) -> PlaceCandidate:
        term = str(payload["term"])
        start = end_index - len(term) + 1
        end = end_index + 1
        return PlaceCandidate(
            text=term,
            canonical_name=str(payload["canonical_name"]),
            place_type=str(payload["place_type"]),
            start=start,
            end=end,
            match_type="exact",
            priority=int(payload["priority"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    @staticmethod
    def _find_generic_facilities(query: str) -> list[PlaceCandidate]:
        candidates: list[PlaceCandidate] = []
        for match in _GENERIC_FACILITY_PATTERN.finditer(query):
            text = match.group("name")
            start = match.start("name")
            text, start = PlaceIndex._normalize_generic_candidate_text(
                text=text,
                start=start,
            )
            if not text:
                continue
            candidates.append(
                PlaceCandidate(
                    text=text,
                    canonical_name=text,
                    place_type="generic_facility",
                    start=start,
                    end=match.end("name"),
                    match_type="generic_facility",
                    priority=_PLACE_TYPE_PRIORITY["generic_facility"],
                )
            )
        return candidates

    @staticmethod
    def _find_region_transport_hubs(
        query: str,
        candidates: list[PlaceCandidate],
    ) -> list[PlaceCandidate]:
        """基于城市候选补充“杭州东/湖州站/上海虹桥”等交通枢纽候选。"""

        transport_candidates: list[PlaceCandidate] = []
        for candidate in candidates:
            if candidate.place_type != "region":
                continue
            for suffix in _REGION_TRANSPORT_SUFFIXES:
                text = f"{candidate.text}{suffix}"
                if query.startswith(text, candidate.start):
                    transport_candidates.append(
                        PlaceCandidate(
                            text=text,
                            canonical_name=text,
                            place_type="generic_transport_hub",
                            start=candidate.start,
                            end=candidate.start + len(text),
                            match_type="generic_transport_hub",
                            priority=_PLACE_TYPE_PRIORITY["generic_transport_hub"],
                            metadata={"base_region": candidate.canonical_name},
                        )
                    )
        return transport_candidates

    @staticmethod
    def _find_generic_roads(query: str) -> list[PlaceCandidate]:
        """补充道路类候选，例如 G60沪昆高速、绕城西线、申嘉湖高速。"""

        candidates: list[PlaceCandidate] = []
        for match in _GENERIC_ROAD_PATTERN.finditer(query):
            text = match.group("name")
            start = match.start("name")
            text, start = PlaceIndex._normalize_generic_candidate_text(
                text=text,
                start=start,
            )
            if len(text) < 2:
                continue
            candidates.append(
                PlaceCandidate(
                    text=text,
                    canonical_name=text,
                    place_type="generic_road",
                    start=start,
                    end=match.end("name"),
                    match_type="generic_road",
                    priority=_PLACE_TYPE_PRIORITY["generic_road"],
                )
            )
        return candidates

    @staticmethod
    def _strip_relation_prefix(*, text: str, start: int) -> tuple[str, int]:
        """泛化候选若吃到连接词，保留最后一个连接词后的地点片段。"""

        relation_matches = list(_RELATION_PATTERN.finditer(text))
        if not relation_matches:
            return text, start
        relation_match = relation_matches[-1]
        stripped_text = text[relation_match.end() :]
        return stripped_text, start + relation_match.end()

    @staticmethod
    def _normalize_generic_candidate_text(*, text: str, start: int) -> tuple[str, int]:
        """标准化泛化候选文本。

        泛化候选依赖正则形态识别，可能会把“请问/今天/查一下/到达”等句首噪声一起吃进来。
        这里先按 OD 连接词切边界，再剥离非地点口语前缀，最后复用端点清洗函数处理尾部噪声。
        """

        text, start = PlaceIndex._strip_relation_prefix(text=text, start=start)
        text, start = PlaceIndex._strip_candidate_prefix(text=text, start=start)

        cleaned_text = clean_endpoint(text)
        if cleaned_text and cleaned_text != text:
            offset = text.rfind(cleaned_text)
            if offset >= 0:
                start += offset
            else:
                start += len(text) - len(cleaned_text)
            text = cleaned_text

        text, start = PlaceIndex._strip_candidate_prefix(text=text, start=start)
        return text, start

    @staticmethod
    def _strip_candidate_prefix(*, text: str, start: int) -> tuple[str, int]:
        """剥离泛化候选前面的口语/时间/查询噪声，并同步修正起始下标。"""

        while True:
            match = CANDIDATE_PREFIX_PATTERN.match(text)
            if not match:
                return text, start
            stripped_text = text[match.end() :]
            if not stripped_text:
                return text, start
            text = stripped_text
            start += match.end()

    @staticmethod
    def _deduplicate_overlaps(candidates: list[PlaceCandidate]) -> list[PlaceCandidate]:
        """重叠候选优先保留更长候选，再比较类型优先级。"""

        ordered = sorted(
            candidates,
            key=lambda item: (item.end - item.start, item.priority, -item.start),
            reverse=True,
        )
        selected: list[PlaceCandidate] = []
        for candidate in ordered:
            if any(
                candidate.start < existing.end and existing.start < candidate.end
                for existing in selected
            ):
                continue
            selected.append(candidate)
        return sorted(selected, key=lambda item: (item.start, item.end))

    @staticmethod
    def _string_items(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item or "").strip()]

    @staticmethod
    def _unique_terms(values: list[str]) -> list[str]:
        seen: set[str] = set()
        terms: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            terms.append(value)
        return terms


@lru_cache(maxsize=1)
def load_default_place_index() -> PlaceIndex:
    """加载默认地点候选索引。"""

    return PlaceIndex()
