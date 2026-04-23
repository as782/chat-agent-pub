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


def test_filter_road_payload_keeps_all_items_without_semantic_direction() -> None:
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

    assert [item["id"] for item in filtered["congestionInfoList"]] == ["up", "down"]


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
