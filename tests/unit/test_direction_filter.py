"""方向过滤辅助模块单元测试。"""

from app.agent.direction_filter import (
    filter_road_payload_events_for_travel_direction,
    filter_section_events_for_travel_direction,
)


def test_filter_section_events_prefers_destination_direction() -> None:
    section = {
        "roadName": "沪昆高速",
        "trafficCongestions": [
            {"id": "hz", "directionType": "02", "des": "G60沪昆高速-杭州方向，缓行"},
            {"id": "jx", "directionType": "01", "des": "G60沪昆高速-江西方向，缓行"},
            {"id": "bi", "directionType": "00", "des": "G60沪昆高速-双向，施工"},
        ],
        "trafficControls": [
            {"id": "tc-hz", "directionType": "02", "des": "G60沪昆高速-杭州方向，抛锚"},
            {"id": "tc-jx", "directionType": "01", "des": "G60沪昆高速-江西方向，事故"},
            {"id": "tc-bi", "directionType": "00", "des": "G60沪昆高速-双向，养护"},
        ],
    }

    filtered = filter_section_events_for_travel_direction(
        section=section,
        origin="杭州",
        destination="义乌",
    )

    assert [item["id"] for item in filtered["trafficCongestions"]] == ["jx", "bi"]
    assert [item["id"] for item in filtered["trafficControls"]] == ["tc-jx", "tc-bi"]


def test_filter_road_payload_drops_one_way_items_without_semantic_direction_for_od_query() -> None:
    payload = {
        "roadName": "杭州绕城高速",
        "congestionInfoList": [
            {"id": "up", "directionType": "01"},
            {"id": "down", "directionType": "02"},
        ],
        "trafficControlList": [],
    }

    filtered = filter_road_payload_events_for_travel_direction(
        road_payload=payload,
        origin="杭州",
        destination="义乌",
    )

    assert filtered["congestionInfoList"] == []


def test_filter_section_events_drops_single_direction_controls_when_only_direction_type_is_available() -> None:
    section = {
        "roadName": "诸永高速",
        "trafficCongestions": [],
        "trafficControls": [
            {"id": "up", "directionType": "01"},
            {"id": "down", "directionType": "02"},
            {"id": "bi", "directionType": "00"},
        ],
    }

    filtered = filter_section_events_for_travel_direction(
        section=section,
        origin="温州",
        destination="磐安",
    )

    assert [item["id"] for item in filtered["trafficControls"]] == ["bi"]


def test_filter_section_events_keeps_single_remaining_direction_type_when_not_ambiguous() -> None:
    section = {
        "roadName": "杭金衢高速",
        "trafficCongestions": [],
        "trafficControls": [
            {"id": "up", "directionType": "01"},
            {"id": "bi", "directionType": "00"},
        ],
    }

    filtered = filter_section_events_for_travel_direction(
        section=section,
        origin="杭州",
        destination="金华",
    )

    assert [item["id"] for item in filtered["trafficControls"]] == ["up", "bi"]


def test_filter_road_payload_still_supports_explicit_up_down_direction_without_semantic_labels() -> None:
    payload = {
        "roadName": "杭州绕城高速",
        "congestionInfoList": [
            {"id": "up", "directionType": "01"},
            {"id": "down", "directionType": "02"},
            {"id": "bi", "directionType": "00"},
        ],
        "trafficControlList": [],
    }

    filtered = filter_road_payload_events_for_travel_direction(
        road_payload=payload,
        origin="杭州",
        destination="义乌",
        explicit_direction="上行",
    )

    assert [item["id"] for item in filtered["congestionInfoList"]] == ["up", "bi"]


def test_filter_road_payload_prefers_explicit_direction_over_od() -> None:
    payload = {
        "roadName": "沪昆高速",
        "congestionInfoList": [
            {"id": "hz", "directionType": "02", "des": "G60沪昆高速-杭州方向，缓行"},
            {"id": "jx", "directionType": "01", "des": "G60沪昆高速-江西方向，缓行"},
            {"id": "bi", "directionType": "00", "des": "G60沪昆高速-双向，施工"},
        ],
        "trafficControlList": [],
    }

    filtered = filter_road_payload_events_for_travel_direction(
        road_payload=payload,
        origin="杭州",
        destination="义乌",
        explicit_direction="杭州方向",
    )

    assert [item["id"] for item in filtered["congestionInfoList"]] == ["hz", "bi"]


def test_filter_section_events_drops_origin_only_semantic_direction_label() -> None:
    section = {
        "roadName": "杭州绕城高速",
        "trafficCongestions": [],
        "trafficControls": [
            {"id": "hz-only", "directionType": "02", "des": "G2504杭州绕城高速-杭州方向，主线管制"},
            {"id": "bi", "directionType": "00", "des": "G2504杭州绕城高速-双向，施工"},
        ],
    }

    filtered = filter_section_events_for_travel_direction(
        section=section,
        origin="杭州",
        destination="温州",
    )

    assert [item["id"] for item in filtered["trafficControls"]] == ["bi"]


def test_filter_section_events_drops_documented_event_types() -> None:
    section = {
        "roadName": "G60",
        "trafficCongestions": [
            {"id": "traffic-event", "eventType": "01", "directionType": "00"},
            {"id": "vehicle-fault", "eventType": "97", "directionType": "00"},
            {"id": "roadwork", "eventType": "05", "directionType": "00"},
        ],
        "trafficControls": [
            {"id": "control-traffic-event", "eventType": "01", "directionType": "00"},
            {"id": "control-roadwork", "eventType": "05", "directionType": "00"},
        ],
    }

    filtered = filter_section_events_for_travel_direction(
        section=section,
        origin="杭州",
        destination="金华",
    )

    assert [item["id"] for item in filtered["trafficCongestions"]] == ["roadwork"]
    assert [item["id"] for item in filtered["trafficControls"]] == ["control-roadwork"]


def test_filter_road_payload_events_drops_documented_event_types() -> None:
    payload = {
        "roadName": "G25",
        "congestionInfoList": [
            {"id": "vehicle-fault", "eventType": "97", "directionType": "00"},
            {"id": "congestion", "eventType": "03", "directionType": "00"},
        ],
        "trafficControlList": [
            {"id": "traffic-event", "eventType": "01", "directionType": "00"},
            {"id": "construction", "eventType": "05", "directionType": "00"},
        ],
    }

    filtered = filter_road_payload_events_for_travel_direction(
        road_payload=payload,
        origin=None,
        destination=None,
    )

    assert [item["id"] for item in filtered["congestionInfoList"]] == ["congestion"]
    assert [item["id"] for item in filtered["trafficControlList"]] == ["construction"]
