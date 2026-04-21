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
                            "endMilestone": 128,
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
                            "endMilestone": 124,
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
    assert "整体判断： 当前状态：拥堵/缓行与交通管制并存" in traffic_context
    assert "拥堵/缓行：" in traffic_context
    assert "K120~K128（上行）：金华方向缓行 | 缓行8公里 | 2026-04-15 08:00:00~2026-04-15 10:30:00" in traffic_context
    assert "交通管制列表：" in traffic_context
    assert "K122~K124（上行）：施工占道 | 2026-04-15 07:30:00~2026-04-15 18:00:00 | 管制措施：封闭第一车道" in traffic_context
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
