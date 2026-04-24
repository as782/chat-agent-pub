from __future__ import annotations

from json import dumps

import pytest

from app.agent.nodes.route_node import RouteNode
from app.agent.state import ExecutionPlan, ResolvedArguments


class _RouteTemplateRegistry:
    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name != "live_driving_query":
            raise AssertionError(f"unexpected tool: {tool_name}")

        return dumps(
            {
                "routesCount": 2,
                "routes": [
                    {
                        "distance": 180000,
                        "duration": 120,
                        "toll": 85,
                        "tags": ["推荐", "高速优先"],
                        "sections": [
                            {
                                "roadName": "杭金衢高速",
                                "exitInfos": [
                                    {
                                        "tollName": "诸暨北收费站",
                                        "entranceStatus": 0,
                                        "exportStatus": 10203,
                                    }
                                ],
                                "serviceAreas": [
                                    {"serviceName": "诸暨服务区", "directionType": "1"}
                                ],
                                "trafficCongestions": [
                                    {
                                        "id": "cg-1",
                                        "beginMilestone": 120,
                                        "endMilestone": 128,
                                        "directionType": "1",
                                        "des": "金华方向缓行",
                                        "beginTime": "2026-04-15 08:00:00",
                                        "controlMeasures": "借道通行",
                                    }
                                ],
                                "trafficControls": [
                                    {
                                        "beginMilestone": 120,
                                        "endMilestone": 128,
                                        "directionType": "1",
                                        "des": "金华方向缓行",
                                        "beginTime": "2026-04-15 08:00:00",
                                        "expectedEndTime": "2026-04-15 10:30:00",
                                        "controlMeasures": "借道通行",
                                    }
                                ],
                            },
                            {
                                "roadName": "沪昆高速",
                                "serviceAreas": [],
                                "trafficControls": [],
                            },
                        ],
                    },
                    {
                        "distance": 195000,
                        "duration": 135,
                        "toll": 78,
                        "tags": ["备选"],
                        "sections": [
                            {
                                "roadName": "杭州绕城高速",
                                "serviceAreas": [
                                    {"serviceName": "金华服务区", "directionType": "2"}
                                ],
                                "trafficControls": [],
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_route_node_builds_template_style_route_context() -> None:
    node = RouteNode(tool_registry=_RouteTemplateRegistry())

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="route_planning",
                execution_mode="single_step",
                recommended_route="route",
            ),
            "resolved_arguments": ResolvedArguments(
                category="route_planning",
                arguments={
                    "origin": "杭州",
                    "destination": "金华",
                    "travel_mode": "auto",
                },
            ),
        }
    )

    route_context = result["route_context"]
    assert route_context is not None
    assert "查询参数：起点：杭州" in route_context
    assert "终点：金华" in route_context
    assert "共查询路线方案（共 2 条）" in route_context
    assert "方案 1 [推荐、高速优先]：路线共180km | 预计耗时2小时 | 费用过路费85元" in route_context
    assert "途经路段：杭金衢高速 → 沪昆高速" in route_context
    assert "诸暨服务区（上行）" in route_context
    assert "K120-K128（上行）：金华方向缓行 | 2026-04-15 08:00:00-2026-04-15 10:30:00 | 管制措施：借道通行" in route_context
    assert "诸暨北收费站（入口开启 / 出口限流）" in route_context
    assert "K120-K128（上行）：金华方向缓行 | 开始时间：2026-04-15 08:00:00 | 管制措施：借道通行" in route_context
    assert "方案 2 [备选]：路线共195km | 预计耗时2小时15分 | 费用过路费78元" in route_context
    assert "杭州绕城高速" in route_context
    assert "金华服务区（下行）" in route_context
