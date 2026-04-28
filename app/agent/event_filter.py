"""Event filtering helpers shared by traffic, route, and report flows."""

from __future__ import annotations

FILTERED_EVENT_CLASSES = {"04", "07"}
FILTERED_EVENT_TYPES = {"01", "97"}
FILTERED_CONTROL_TYPES = {"10110"}


def should_filter_live_event(item: dict[str, object]) -> bool:
    """Return True when a live route/road event should be dropped by eventType."""

    return _first_non_empty(item.get("eventType")) in FILTERED_EVENT_TYPES


def should_filter_report_event(item: dict[str, object]) -> bool:
    """Return True when a network report event should be dropped."""

    event_class = _first_non_empty(item.get("eventClass"))
    event_type = _first_non_empty(item.get("eventType"))
    control_type = _first_non_empty(item.get("controlType"))
    return (
        event_class in FILTERED_EVENT_CLASSES
        or event_type in FILTERED_EVENT_TYPES
        or control_type in FILTERED_CONTROL_TYPES
    )


def _first_non_empty(value: object) -> str:
    return str(value or "").strip()
