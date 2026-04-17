"""Helpers for parsing and normalizing traffic road identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_ROAD_CODE_PATTERN = re.compile(r"^[GS]\d{1,4}$", re.IGNORECASE)
_ROAD_CODE_SEARCH_PATTERN = re.compile(r"([GS]\d{1,4})(?!\d)", re.IGNORECASE)
_DIRECTION_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,8}\u65b9\u5411|\u4e0a\u884c|\u4e0b\u884c)"
)
_DIRECTION_ALIAS_PATTERN = re.compile(
    r"([\u4e1c\u897f\u5357\u5317\u6caa\u676d\u752c\u5b81\u6e29\u82cf\u5609\u6e56\u7ecd\u91d1\u8862\u53f0\u4e3d]{1,3}\u5411)"
)
_TOLL_STATION_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{2,20}(?:\u6536\u8d39\u7ad9|\u6536\u8d39\u53e3))"
)
_ROAD_NAME_HINTS = (
    "\u9ad8\u901f",
    "\u9ad8\u901f\u516c\u8def",
    "\u7ed5\u57ce",
    "\u73af\u7ebf",
    "\u5feb\u901f\u8def",
    "\u56fd\u9053",
    "\u7701\u9053",
    "\u5927\u9053",
    "\u5927\u6865",
    "\u96a7\u9053",
    "\u8def\u6bb5",
)
_ROAD_SEPARATORS = " \t,\uff0c:\uff1a;\uff1b/\\-_|()\uff08\uff09[]\u3010\u3011"


@dataclass(frozen=True, slots=True)
class InferredRoadContext:
    road: str | None = None
    roads: tuple[str, ...] = ()
    target: str | None = None
    direction: str | None = None
    toll_station: str | None = None


def split_road_identifier(
    value: str,
    *,
    trust_name_field: bool = False,
) -> tuple[str | None, str | None]:
    """Split a road identifier into pure code and pure name when possible."""

    normalized_value = str(value or "").strip()
    if not normalized_value:
        return None, None

    code_match = _ROAD_CODE_SEARCH_PATTERN.search(normalized_value)
    road_code = code_match.group(1).upper() if code_match is not None else None

    if road_code is not None:
        road_name_candidate = _ROAD_CODE_SEARCH_PATTERN.sub(" ", normalized_value, count=1)
        road_name_candidate = road_name_candidate.strip(_ROAD_SEPARATORS)
        road_name_candidate = " ".join(road_name_candidate.split())
        if road_name_candidate and not _ROAD_CODE_PATTERN.fullmatch(road_name_candidate):
            return road_code, road_name_candidate
        return road_code, None

    if trust_name_field or _looks_like_road_name(normalized_value):
        return None, normalized_value
    return None, None


def normalize_road_query_value(
    value: str,
    *,
    prefer: Literal["code", "name"] = "name",
) -> str:
    """Normalize a road query value to a single code-or-name identifier."""

    normalized_value = str(value or "").strip()
    if not normalized_value:
        return ""

    road_code, road_name = split_road_identifier(normalized_value)
    if prefer == "code":
        return road_code or road_name or normalized_value
    return road_name or road_code or normalized_value


def normalize_road_query_list(
    values: list[str] | tuple[str, ...],
    *,
    prefer: Literal["code", "name"] = "name",
) -> list[str]:
    """Normalize a road list while preserving order and uniqueness."""

    normalized_roads: list[str] = []
    seen_roads: set[str] = set()
    for raw_value in values:
        normalized_value = normalize_road_query_value(str(raw_value), prefer=prefer)
        if not normalized_value or normalized_value in seen_roads:
            continue
        seen_roads.add(normalized_value)
        normalized_roads.append(normalized_value)
    return normalized_roads


def normalize_traffic_road_fields(
    *,
    road: object = None,
    road_name: object = None,
    road_code: object = None,
    prefer: Literal["code", "name"] = "name",
) -> dict[str, str]:
    """Normalize traffic road fields so each field holds a single identifier form."""

    raw_road = str(road or "").strip()
    raw_road_name = str(road_name or "").strip()
    raw_road_code = str(road_code or "").strip()

    code_from_name, normalized_road_name = split_road_identifier(
        raw_road_name,
        trust_name_field=True,
    )
    code_from_code, _ = split_road_identifier(raw_road_code)
    code_from_road, normalized_name_from_road = split_road_identifier(raw_road)

    normalized_code = code_from_code or code_from_name or code_from_road
    normalized_name = normalized_road_name or normalized_name_from_road

    if prefer == "code":
        normalized_road = normalized_code or normalized_name or raw_road or raw_road_name or raw_road_code
    else:
        normalized_road = normalized_name or normalized_code or raw_road or raw_road_name or raw_road_code

    normalized_fields: dict[str, str] = {}
    if normalized_road:
        normalized_fields["road"] = normalized_road
    if normalized_name:
        normalized_fields["road_name"] = normalized_name
    if normalized_code:
        normalized_fields["road_code"] = normalized_code
    return normalized_fields


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
        match = _DIRECTION_ALIAS_PATTERN.search(normalized_text)
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
    normalized_roads = normalize_road_query_list(list(explicit_roads or []), prefer="name")
    primary_road = normalized_roads[0] if normalized_roads else None

    toll_station = extract_toll_station(candidate_text) or extract_toll_station(message)
    direction_source = candidate_text
    if toll_station:
        direction_source = direction_source.replace(toll_station, " ")

    direction = extract_direction(direction_source) or extract_direction(message)

    return InferredRoadContext(
        road=primary_road,
        roads=tuple(normalized_roads),
        target=candidate_text or None,
        direction=direction,
        toll_station=toll_station,
    )


def _looks_like_road_name(value: str) -> bool:
    normalized_value = value.strip()
    return any(token in normalized_value for token in _ROAD_NAME_HINTS)
