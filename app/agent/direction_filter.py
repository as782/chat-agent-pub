"""Helpers for filtering road-level aggregate events by OD travel direction."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.agent.event_filter import should_filter_live_event

_DIRECTION_HINT_PATTERN = re.compile(
    r"(双向|上行|下行|[东南西北]{1,2}向|[\u4e00-\u9fffA-Za-z0-9]{1,12}方向)"
)
_PLACE_SUFFIX_PATTERN = re.compile(
    r"(省|市|地区|自治州|州|盟|县|区|镇|乡|街道|收费站|收费口|服务区|方向|向)$"
)


def filter_section_events_for_travel_direction(
    *,
    section: dict[str, object],
    origin: object,
    destination: object,
    explicit_direction: object = None,
) -> dict[str, object]:
    """Filter a route section's congestion/control items by travel direction."""

    filtered_section = dict(section)
    filtered_section["trafficCongestions"] = _filter_directional_items(
        section.get("trafficCongestions"),
        origin=origin,
        destination=destination,
        explicit_direction=explicit_direction,
    )
    filtered_section["trafficControls"] = _filter_directional_items(
        section.get("trafficControls"),
        origin=origin,
        destination=destination,
        explicit_direction=explicit_direction,
    )
    return filtered_section


def filter_road_payload_events_for_travel_direction(
    *,
    road_payload: dict[str, object],
    origin: object,
    destination: object,
    explicit_direction: object = None,
) -> dict[str, object]:
    """Filter a road-event payload's congestion/control lists by travel direction."""

    filtered_payload = dict(road_payload)
    filtered_payload["congestionInfoList"] = _filter_directional_items(
        road_payload.get("congestionInfoList"),
        origin=origin,
        destination=destination,
        explicit_direction=explicit_direction,
    )
    filtered_payload["trafficControlList"] = _filter_directional_items(
        road_payload.get("trafficControlList"),
        origin=origin,
        destination=destination,
        explicit_direction=explicit_direction,
    )
    return filtered_payload


def _filter_directional_items(
    raw_items: object,
    *,
    origin: object,
    destination: object,
    explicit_direction: object,
) -> list[dict[str, object]]:
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    items = [item for item in items if not should_filter_live_event(item)]
    if not items:
        return []

    directional_context = _build_directional_context(
        items=items,
        origin=origin,
        destination=destination,
        explicit_direction=explicit_direction,
    )
    if not directional_context:
        return items

    return [item for item in items if _should_keep_directional_item(item, directional_context)]


def _build_directional_context(
    *,
    items: list[dict[str, object]],
    origin: object,
    destination: object,
    explicit_direction: object,
) -> dict[str, object] | None:
    semantic_labels = _collect_semantic_labels(items)
    if not semantic_labels:
        explicit_direction_type = _normalize_direction_type(explicit_direction)
        if explicit_direction_type and explicit_direction_type != "00":
            return {
                "allowed_labels": set(),
                "allowed_types": {explicit_direction_type},
                "drop_unmatched_directional": False,
            }

        non_bidirectional_types = _collect_non_bidirectional_direction_types(items)
        if len(non_bidirectional_types) == 1:
            return {
                "allowed_labels": set(),
                "allowed_types": set(non_bidirectional_types),
                "drop_unmatched_directional": False,
            }
        if non_bidirectional_types and _has_directional_query_context(
            origin=origin,
            destination=destination,
            explicit_direction=explicit_direction,
        ):
            return {
                "allowed_labels": set(),
                "allowed_types": set(),
                "drop_unmatched_directional": True,
            }
        return None

    explicit_matches = _match_semantic_labels(explicit_direction, semantic_labels)
    destination_matches = _match_semantic_labels(destination, semantic_labels)
    origin_matches = _match_semantic_labels(origin, semantic_labels)

    if explicit_matches:
        allowed_labels = explicit_matches
    elif destination_matches:
        allowed_labels = destination_matches
    elif origin_matches:
        opposite_labels = semantic_labels - origin_matches
        if opposite_labels:
            allowed_labels = opposite_labels
        else:
            return {
                "allowed_labels": set(),
                "allowed_types": set(),
                "drop_unmatched_directional": True,
            }
    else:
        allowed_labels = set()

    if not allowed_labels:
        return None

    allowed_types = _resolve_allowed_direction_types(items, allowed_labels)
    return {
        "allowed_labels": allowed_labels,
        "allowed_types": allowed_types,
        "drop_unmatched_directional": False,
    }


def _collect_semantic_labels(items: Iterable[dict[str, object]]) -> set[str]:
    semantic_labels: set[str] = set()
    for item in items:
        direction_hint = _extract_direction_hint(item)
        if direction_hint in {None, "双向", "上行", "下行"}:
            continue
        semantic_labels.add(direction_hint)
    return semantic_labels


def _collect_non_bidirectional_direction_types(items: Iterable[dict[str, object]]) -> set[str]:
    direction_types: set[str] = set()
    for item in items:
        direction_type = _normalize_direction_type(item.get("directionType"))
        if direction_type in {None, "00"}:
            continue
        direction_types.add(direction_type)
    return direction_types


def _has_directional_query_context(
    *,
    origin: object,
    destination: object,
    explicit_direction: object,
) -> bool:
    return any(str(value or "").strip() for value in (origin, destination, explicit_direction))


def _resolve_allowed_direction_types(
    items: Iterable[dict[str, object]],
    allowed_labels: set[str],
) -> set[str]:
    allowed_types: set[str] = set()
    for item in items:
        direction_hint = _extract_direction_hint(item)
        if direction_hint not in allowed_labels:
            continue
        direction_type = _normalize_direction_type(item.get("directionType"))
        if direction_type:
            allowed_types.add(direction_type)
    return allowed_types


def _should_keep_directional_item(
    item: dict[str, object],
    directional_context: dict[str, object],
) -> bool:
    direction_type = _normalize_direction_type(item.get("directionType"))
    direction_hint = _extract_direction_hint(item)

    if direction_type == "00" or direction_hint == "双向":
        return True

    allowed_labels = directional_context.get("allowed_labels")
    if isinstance(allowed_labels, set) and direction_hint in allowed_labels:
        return True

    allowed_types = directional_context.get("allowed_types")
    if isinstance(allowed_types, set) and direction_type in allowed_types:
        return True

    drop_unmatched_directional = directional_context.get("drop_unmatched_directional")
    if drop_unmatched_directional is True:
        return False

    if direction_hint in {None, "上行", "下行"} and not allowed_types:
        return True

    return False


def _extract_direction_hint(item: dict[str, object]) -> str | None:
    for candidate in (
        item.get("directionName"),
        item.get("direction"),
        item.get("directionLabel"),
        item.get("des"),
        item.get("description"),
        item.get("content"),
    ):
        normalized_candidate = _normalize_direction_hint(candidate)
        if normalized_candidate:
            return normalized_candidate
    return _normalize_direction_hint(item.get("directionType"))


def _normalize_direction_hint(value: object) -> str | None:
    raw_text = str(value or "").strip()
    if not raw_text:
        return None

    if raw_text in {"双向", "上行", "下行"}:
        return raw_text

    normalized_type = _normalize_direction_type(raw_text)
    if normalized_type == "00":
        return "双向"
    if normalized_type == "01":
        return "上行"
    if normalized_type == "02":
        return "下行"

    match = _DIRECTION_HINT_PATTERN.search(raw_text)
    if match is None:
        return None
    return match.group(1).strip()


def _normalize_direction_type(value: object) -> str | None:
    raw_text = str(value or "").strip()
    if not raw_text:
        return None

    if raw_text in {"双向", "0", "00"}:
        return "00"
    if raw_text == "上行":
        return "01"
    if raw_text == "下行":
        return "02"

    stripped = raw_text.lstrip("0")
    if stripped == "1":
        return "01"
    if stripped == "2":
        return "02"
    return None


def _match_semantic_labels(value: object, semantic_labels: Iterable[str]) -> set[str]:
    normalized_value = _normalize_place_name(value)
    if not normalized_value:
        return set()

    matches: set[str] = set()
    for label in semantic_labels:
        normalized_label = _normalize_place_name(label)
        if not normalized_label:
            continue
        if normalized_value in normalized_label or normalized_label in normalized_value:
            matches.add(label)
    return matches


def _normalize_place_name(value: object) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""

    while True:
        updated = _PLACE_SUFFIX_PATTERN.sub("", normalized).strip()
        if updated == normalized:
            break
        normalized = updated
    return normalized
