"""Helpers for parsing traffic-query surface context without hardcoded road mappings."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ROAD_CODE_PATTERN = re.compile(r"^[GS]\d{1,4}$", re.IGNORECASE)
_DIRECTION_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,8}方向|[东南西北沪杭甬宁温苏嘉湖绍金台衢丽]{1,3}向|上行|下行)"
)
_TOLL_STATION_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,20}(?:收费站|收费口))")


@dataclass(frozen=True, slots=True)
class InferredRoadContext:
    road: str | None = None
    roads: tuple[str, ...] = ()
    target: str | None = None
    direction: str | None = None
    toll_station: str | None = None


def extract_toll_station(text: str) -> str | None:
    """Extract a toll station phrase from free-form traffic text."""

    match = _TOLL_STATION_PATTERN.search(text.strip())
    return match.group(1).strip() if match is not None else None


def extract_direction(text: str) -> str | None:
    """Extract a direction phrase from free-form traffic text."""

    normalized_text = text.strip()
    if not normalized_text:
        return None

    match = _DIRECTION_PATTERN.search(normalized_text)
    if match is None:
        return None
    return match.group(1).strip()


def infer_traffic_context(
    *,
    message: str,
    normalized_target: str,
    explicit_roads: list[str] | None = None,
) -> InferredRoadContext:
    """Parse traffic query context and keep only non-mapped surface information."""

    candidate_text = normalized_target or message.strip()
    normalized_roads: list[str] = []
    seen_roads: set[str] = set()
    for raw_road in explicit_roads or []:
        normalized_road = raw_road.strip()
        if not normalized_road:
            continue
        if _ROAD_CODE_PATTERN.fullmatch(normalized_road):
            normalized_road = normalized_road.upper()
        if normalized_road in seen_roads:
            continue
        seen_roads.add(normalized_road)
        normalized_roads.append(normalized_road)
    primary_road = normalized_roads[0] if normalized_roads else None

    toll_station = extract_toll_station(candidate_text) or extract_toll_station(message)
    direction_source = candidate_text
    if toll_station:
        direction_source = direction_source.replace(toll_station, " ")

    direction = extract_direction(direction_source) or extract_direction(message)

    road = primary_road

    return InferredRoadContext(
        road=road,
        roads=tuple(normalized_roads),
        target=candidate_text or None,
        direction=direction,
        toll_station=toll_station,
    )
