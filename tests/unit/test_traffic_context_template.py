from __future__ import annotations

from json import dumps

import pytest

from app.agent.nodes.traffic_node import TrafficNode
from app.agent.state import ExecutionPlan, ResolvedArguments


class _TemplateTrafficRegistry:
    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name != "live_road_event_query":
            raise AssertionError(f"unexpected tool: {tool_name}")

        return dumps(
            [
                {
                    "roadName": "杭金衢高速",
                    "roadGbCode": "G60",
                    "congestionInfoList": [
                        {
                            "des": "金华方向缓行",
                            "beginMilestone": 120,
                            "beginMilestoneStr": "K120+0",
                            "endMilestone": 128,
                            "endMilestoneStr": "K128+0",
                            "directionType": "1",
                            "beginTime": "2026-04-15 08:00:00",
                            "expectedEndTime": "2026-04-15 10:30:00",
                            "roadAmbleMile": 8.0,
                        }
                    ],
                    "trafficControlList": [
                        {
                            "des": "施工占道",
                            "beginMilestone": 122,
                            "beginMilestoneStr": "K122+0",
                            "endMilestone": 124,
                            "endMilestoneStr": "K124+0",
                            "directionType": "1",
                            "beginTime": "2026-04-15 07:30:00",
                            "expectedEndTime": "2026-04-15 18:00:00",
                            "controlMeasures": "封闭第一车道",
                        }
                    ],
                    "serviceAreaList": [
                        {
                            "serviceName": "杭州服务区",
                            "directionType": "1",
                            "statusTag": "繁忙",
                        }
                    ],
                    "exitInfoList": [
                        {
                            "tollName": "杭州北",
                            "entranceStatus": 10202,
                            "exportStatus": 10203,
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_traffic_node_builds_template_style_traffic_context() -> None:
    node = TrafficNode(tool_registry=_TemplateTrafficRegistry())

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="traffic_status",
                execution_mode="single_step",
                recommended_route="traffic",
            ),
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={"target": "杭金衢高速", "time_range": "current"},
            ),
        }
    )

    traffic_context = result["traffic_context"]
    assert traffic_context is not None
    assert "道路名称：杭金衢高速 编号：G60，路况如下：" in traffic_context
    assert "整体统计：拥堵/缓行事件：1条 ,交通管制事件：1条 , 异常收费站：1个 ,状态异常服务区 ：1个" in traffic_context
    assert "拥堵/缓行列表：" in traffic_context
    assert "K120+0-K128+0（上行）：金华方向缓行 | 缓行8公里 | 2026-04-15 08:00:00-2026-04-15 10:30:00" in traffic_context
    assert "交通管制列表：" in traffic_context
    assert "K122+0-K124+0（上行）：施工占道 | 2026-04-15 07:30:00-2026-04-15 18:00:00 | 管制措施：封闭第一车道" in traffic_context
    assert "收费站列表：" in traffic_context
    assert "杭州北：入口关闭 / 出口限流" in traffic_context
    assert "服务区列表：" in traffic_context
    assert "杭州服务区（上行）：繁忙" in traffic_context


def test_station_status_label_normalizes_int_and_string_codes() -> None:
    assert TrafficNode._resolve_station_status_label(0) == "开启"
    assert TrafficNode._resolve_station_status_label("0") == "开启"
    assert TrafficNode._resolve_station_status_label(10202) == "关闭"
    assert TrafficNode._resolve_station_status_label("10202") == "关闭"
    assert TrafficNode._resolve_station_status_label(" 10203 ") == "限流"
    assert TrafficNode._normalize_status_code(" 1233 ") == "1233"
    assert TrafficNode._normalize_status_code(1233) == "1233"
    assert TrafficNode._is_abnormal_station_status("0") is False
    assert TrafficNode._is_abnormal_station_status("10202") is True


def test_traffic_control_items_filter_vehicle_fault_events() -> None:
    traffic_control_items = TrafficNode._extract_traffic_control_items(
        [
            {
                "roadName": "S26诸永高速",
                "roadGbCode": "S26",
                "trafficControlList": [
                    {
                        "id": "vehicle-fault",
                        "eventType": "97",
                        "subEventType": "970100",
                        "des": "车辆故障，占据二车道、硬路肩",
                    },
                    {
                        "id": "construction",
                        "eventType": "05",
                        "subEventType": "050101",
                        "des": "道路施工，占据硬路肩",
                    },
                ],
            }
        ]
    )

    assert [item["control_id"] for item in traffic_control_items] == ["construction"]


def test_congestion_items_filter_vehicle_fault_events() -> None:
    congestion_items = TrafficNode._extract_congestion_items(
        [
            {
                "roadName": "S26诸永高速",
                "roadGbCode": "S26",
                "congestionInfoList": [
                    {
                        "id": "vehicle-fault",
                        "eventType": "97",
                        "subEventType": "970100",
                        "des": "车辆故障，占据二车道、硬路肩",
                    },
                    {
                        "id": "slow",
                        "eventType": "105",
                        "des": "道路缓行",
                    },
                ],
            }
        ]
    )

    assert [item["congestion_id"] for item in congestion_items] == ["slow"]


def test_traffic_context_lists_all_items_without_local_truncation() -> None:
    road = {
        "roadName": "S26诸永高速",
        "roadGbCode": "S26",
        "trafficControlList": [
            {
                "id": f"construction-{index}",
                "eventType": "05",
                "subEventType": "050101",
                "directionType": "00",
                "beginMilestone": index,
                "endMilestone": index + 1,
                "des": f"道路施工{index}",
            }
            for index in range(4)
        ],
        "congestionInfoList": [
            {
                "id": f"slow-{index}",
                "eventType": "105",
                "directionType": "00",
                "beginMilestone": index,
                "endMilestone": index + 1,
                "des": f"道路缓行{index}",
            }
            for index in range(4)
        ],
        "serviceAreaList": [
            {
                "serviceName": f"服务区{index}",
                "directionType": "00",
                "statusTag": "正常",
            }
            for index in range(4)
        ],
        "exitInfoList": [
            {
                "tollName": f"收费站{index}",
                "entranceStatus": 0,
                "exportStatus": 0,
            }
            for index in range(4)
        ],
    }

    context = TrafficNode._build_compact_traffic_road_block(
        road=road,
        query_arguments={"query": "S26路况怎么样"},
    )

    for index in range(4):
        assert f"道路施工{index}" in context
        assert f"道路缓行{index}" in context
        assert f"服务区{index}" in context
        assert f"收费站{index}" in context


def test_focus_query_block_includes_matched_toll_station_status_and_related_controls() -> None:
    response_payload = [
        {
            "roadName": "G1512甬金高速（金华段）",
            "roadGbCode": "G1512",
            "trafficControlList": [
                {
                    "des": "G1512甬金高速（金华段）发生协助处理，佛堂收费站宁波方向出口分流",
                    "directionType": "02",
                    "beginTime": "2026-04-24 10:00:00",
                }
            ],
            "exitInfoList": [
                {
                    "tollName": "蔡宅收费站",
                    "entranceStatus": 0,
                    "exportStatus": 0,
                },
                {
                    "tollName": "佛堂收费站",
                    "entranceStatus": 0,
                    "exportStatus": 10204,
                },
            ],
        }
    ]

    focus_block = TrafficNode._build_focus_query_block(
        query_arguments={
            "toll_station": "佛堂收费站",
            "direction": "宁波方向",
        },
        response_payload=response_payload,
    )

    assert focus_block is not None
    assert "收费站：佛堂收费站" in focus_block
    assert "用户关注方向/部位：宁波方向" in focus_block
    assert "佛堂收费站（G1512甬金高速（金华段））：入口开启 / 出口分流" in focus_block
    assert "佛堂收费站宁波方向出口分流" in focus_block


def test_prioritize_exit_items_moves_target_station_to_front() -> None:
    exit_items = [
        {"toll_name": "蔡宅收费站", "entrance_status": 0, "export_status": 0},
        {"toll_name": "佛堂收费站", "entrance_status": 0, "export_status": 10204},
        {"toll_name": "东阳收费站", "entrance_status": 0, "export_status": 0},
    ]

    prioritized_items = TrafficNode._prioritize_exit_items(
        exit_items,
        query_arguments={"toll_station": "佛堂站口"},
    )

    assert prioritized_items[0]["toll_name"] == "佛堂收费站"
