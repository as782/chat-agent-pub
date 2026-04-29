"""本地 OD 起终点解析器。

解析器采用“结构化 OD 句式 + 端点清洗 + 设施目录辅助归一化”的方式，
不要求所有地点都在本地词典内。这样既能利用内部服务区/收费站数据，
也能把省外或未知 POI 作为弱地点传给具备宽松解析能力的路线接口。
"""

from __future__ import annotations

import re

from app.agent.facility_catalog import (
    ServiceAreaRecord,
    TollStationRecord,
    load_default_facility_catalog,
)
from app.agent.od.guard import is_valid_endpoint
from app.agent.od.models import OdResolution, PlaceCandidate
from app.agent.od.normalizer import clean_endpoint, normalize_query
from app.agent.od.place_index import load_default_place_index
from app.core.logger import get_logger

LOGGER = get_logger(__name__)

# 起终点候选允许中文、英文、数字和少量地点常见符号；长度上限避免整句被吞进去。
_PLACE_CHARS = r"[\u4e00-\u9fffA-Za-z0-9·\-（）()]{1,40}"
# 目的地后面的停止条件。遇到“多久/怎么走/堵不堵”等意图词时停止截取地点。
_END_LOOKAHEAD = (
    r"(?=方向|还有|还要|要|多久|多少时间|多长时间|需要|能到|到达|怎么|如何|"
    r"堵不堵|堵吗|路况|收费|过路费|通行费|，|,|。|？|\?|!|！|$)"
)
_DESTINATION_ANCHOR_PATTERN = re.compile(
    rf"(?:能到|到达|到)(?P<destination>{_PLACE_CHARS}?){_END_LOOKAHEAD}"
)
_OD_CONNECTOR_PATTERN = re.compile(r"(?:去往|前往|开往|开到|到|至|往|去|回)")

# OD 结构按优先级排列：
# 1. “我在 X，到/能到 Y”这类当前位置逗号表达；
# 2. “我现在在 X 往 Y”这类当前位置表达；
# 3. “从 X 到 Y”显式起终点；
# 4. “X 到/去/往 Y”通用兜底。
_OD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"(?:我)?(?:现在|目前|当前|此刻)?(?:正?在|位于|处在)"
        rf"(?P<origin>{_PLACE_CHARS}?)[，,](?:能到|到达|到)"
        rf"(?P<destination>{_PLACE_CHARS}?){_END_LOOKAHEAD}"
    ),
    re.compile(
        rf"(?:我)?(?:现在|目前|当前|此刻|已经)?(?:正?在|位于|处在|到达)"
        rf"(?P<origin>{_PLACE_CHARS}?)(?:去往|前往|开往|开到|往|到|去)"
        rf"(?P<destination>{_PLACE_CHARS}?){_END_LOOKAHEAD}"
    ),
    re.compile(
        rf"(?:从|由)(?P<origin>{_PLACE_CHARS}?)(?:去往|前往|开往|开到|到|至|往|去|回)"
        rf"(?P<destination>{_PLACE_CHARS}?){_END_LOOKAHEAD}"
    ),
    re.compile(
        rf"(?P<origin>{_PLACE_CHARS}?)(?:去往|前往|开往|开到|到|至|往|去|回)"
        rf"(?P<destination>{_PLACE_CHARS}?){_END_LOOKAHEAD}"
    ),
)


class OdResolver:
    """本地 OD 解析器。

    目标是输出干净的起终点边界，而不是证明地点一定真实存在。
    """

    def resolve(self, message: str) -> OdResolution:
        """从用户原始问题中解析 OD 参数。"""

        query = normalize_query(message)
        warnings: list[str] = []
        candidates = load_default_place_index().find_candidates(query)
        candidate_resolution = self._resolve_from_candidates(query=query, candidates=candidates)
        if candidate_resolution is not None:
            return candidate_resolution

        for pattern in _OD_PATTERNS:
            match = pattern.search(query)
            if match is None:
                continue

            raw_origin = match.group("origin")
            raw_destination = match.group("destination")
            raw_destination = self._prefer_destination_anchor(
                query=query,
                raw_destination=raw_destination,
                destination_start=match.start("destination"),
            )
            origin = clean_endpoint(raw_origin)
            destination = clean_endpoint(raw_destination)
            # 只有端点本身像服务区/收费站时才走设施归一化，避免“杭州”误归一成收费站。
            origin, origin_match_type = self._normalize_facility_endpoint(origin, query=query)
            destination, destination_match_type = self._normalize_facility_endpoint(
                destination,
                query=query,
            )
            if not origin or not destination:
                warnings.append("empty_endpoint_after_cleanup")
                continue
            if origin == destination:
                warnings.append("same_origin_destination")
                continue
            if not is_valid_endpoint(origin):
                warnings.append("invalid_origin")
                continue
            if not is_valid_endpoint(destination):
                warnings.append("invalid_destination")
                continue

            confidence = self._score_resolution(
                origin_match_type=origin_match_type,
                destination_match_type=destination_match_type,
            )
            resolution = OdResolution(
                origin=origin,
                destination=destination,
                confidence=confidence,
                source="local_od_resolver",
                raw_origin=raw_origin,
                raw_destination=raw_destination,
                origin_match_type=origin_match_type,
                destination_match_type=destination_match_type,
                warnings=warnings,
                candidates=candidates,
            )
            LOGGER.info(
                (
                    "OD resolver result: origin=%s destination=%s confidence=%.2f "
                    "source=%s warnings=%s"
                ),
                resolution.origin,
                resolution.destination,
                resolution.confidence,
                resolution.source,
                resolution.warnings,
            )
            return resolution

        return OdResolution(
            origin=None,
            destination=None,
            confidence=0.0,
            source="local_od_resolver",
            warnings=warnings or ["no_od_pattern_match"],
            candidates=candidates,
        )

    @staticmethod
    def _resolve_from_candidates(
        *,
        query: str,
        candidates: list[PlaceCandidate],
    ) -> OdResolution | None:
        """基于地点候选和 OD 连接词选择起终点。

        这条路径优先于正则截取，避免“收费”这类意图词把“上海收费站”截短。
        """

        if len(candidates) < 2:
            return None
        best_pair: tuple[int, PlaceCandidate, PlaceCandidate] | None = None
        for connector_match in _OD_CONNECTOR_PATTERN.finditer(query):
            connector_start = connector_match.start()
            connector_end = connector_match.end()
            origin_candidates = [
                candidate for candidate in candidates if candidate.end <= connector_start
            ]
            destination_candidates = [
                candidate for candidate in candidates if candidate.start >= connector_end
            ]
            if not origin_candidates or not destination_candidates:
                continue
            origin = max(
                origin_candidates,
                key=lambda item: (item.end, item.priority, item.end - item.start),
            )
            destination = min(
                destination_candidates,
                key=lambda item: (item.start, -(item.priority), -(item.end - item.start)),
            )
            if origin.canonical_name == destination.canonical_name:
                continue
            score = (
                origin.priority
                + destination.priority
                - abs(origin.end - connector_start)
                - abs(destination.start - connector_end)
            )
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, origin, destination)

        if best_pair is None:
            return None
        _, origin, destination = best_pair
        origin_name = clean_endpoint(origin.canonical_name)
        destination_name = clean_endpoint(destination.canonical_name)
        if not is_valid_endpoint(origin_name) or not is_valid_endpoint(destination_name):
            return None
        resolution = OdResolution(
            origin=origin_name,
            destination=destination_name,
            confidence=0.86,
            source="local_od_resolver:place_index",
            raw_origin=origin.text,
            raw_destination=destination.text,
            origin_match_type=OdResolver._candidate_match_type(origin),
            destination_match_type=OdResolver._candidate_match_type(destination),
            candidates=candidates,
        )
        LOGGER.info(
            (
                "OD resolver result: origin=%s destination=%s confidence=%.2f "
                "source=%s candidates=%s"
            ),
            resolution.origin,
            resolution.destination,
            resolution.confidence,
            resolution.source,
            [
                {
                    "text": candidate.text,
                    "type": candidate.place_type,
                    "start": candidate.start,
                    "end": candidate.end,
                }
                for candidate in candidates
            ],
        )
        return resolution

    @staticmethod
    def _candidate_match_type(candidate: PlaceCandidate) -> str:
        """把候选类型映射成对外稳定的匹配类型。"""

        if candidate.place_type == "service_area":
            return "facility_service_area"
        if candidate.place_type == "toll_station":
            return "facility_toll_station"
        return candidate.place_type

    @staticmethod
    def _prefer_destination_anchor(
        *,
        query: str,
        raw_destination: str,
        destination_start: int,
    ) -> str:
        """优先使用后文“到/能到/到达 X”中的目的地锚点。

        口语里常出现“往杭州区...能到杭州”这类前半句目的地带噪声、
        后半句重复真实目的地的情况。后文到达锚点通常更可靠。
        """

        for anchor_match in _DESTINATION_ANCHOR_PATTERN.finditer(query):
            if anchor_match.start("destination") <= destination_start:
                continue
            anchored_destination = clean_endpoint(anchor_match.group("destination"))
            if not anchored_destination:
                continue
            if raw_destination.startswith(anchored_destination):
                return anchored_destination
        return raw_destination

    @staticmethod
    def _normalize_facility_endpoint(endpoint: str, *, query: str) -> tuple[str, str]:
        """用内部设施目录归一化服务区/收费站端点。

        未命中或不像设施的端点会以 structured_text 返回，继续允许上游路线接口解析。
        """

        if not endpoint:
            return "", "empty"
        endpoint_looks_like_service_area = any(token in endpoint for token in ("服务区", "停车区"))
        endpoint_looks_like_toll_station = any(
            token in endpoint for token in ("收费站", "收费口", "主站", "副站")
        )
        catalog = load_default_facility_catalog()
        lookup_query = endpoint
        # 服务区优先，因为“服务区到城市”是当前高速场景的高频问法。
        service_matches = (
            catalog.match_service_area(lookup_query, limit=1)
            if endpoint_looks_like_service_area
            else []
        )
        if service_matches:
            record = service_matches[0].record
            if (
                isinstance(record, ServiceAreaRecord)
                and endpoint in query
                and endpoint_looks_like_service_area
            ):
                return record.group_name or record.canonical_name, "facility_service_area"

        toll_matches = (
            catalog.match_toll_station(lookup_query, limit=1)
            if endpoint_looks_like_toll_station
            else []
        )
        if toll_matches:
            record = toll_matches[0].record
            if (
                isinstance(record, TollStationRecord)
                and endpoint in query
                and endpoint_looks_like_toll_station
            ):
                return record.canonical_name, "facility_toll_station"

        return endpoint, "structured_text"

    @staticmethod
    def _score_resolution(*, origin_match_type: str, destination_match_type: str) -> float:
        """根据匹配来源给解析结果一个内部置信度。"""

        score = 0.62
        if origin_match_type.startswith("facility"):
            score += 0.16
        if destination_match_type.startswith("facility"):
            score += 0.12
        return min(score, 0.95)


def resolve_od(message: str) -> OdResolution:
    """便捷函数：使用默认解析器解析一条用户问题。"""

    return OdResolver().resolve(message)
