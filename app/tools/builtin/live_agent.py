"""直播问答接口工具模块。

负责把直播问答接口文档中的标准 HTTP 能力包装为统一工具，供业务节点和通用工具链复用。
当前阶段返回 JSON 字符串结果，方便写入消息历史并在回答节点中继续加工。
"""

from __future__ import annotations

from json import dumps

from langchain_core.tools import tool

from app.clients.live_agent_client import LiveAgentClient


@tool("live_driving_query")
async def live_driving_query(start: str, end: str) -> str:
    """查询两地之间的路线规划信息。"""

    client = LiveAgentClient()
    result = await client.query_driving_plan(start=start, end=end)
    return dumps(result, ensure_ascii=False)


@tool("live_road_event_query")
async def live_road_event_query(road: str) -> str:
    """查询指定道路的路况、拥堵和交通管制信息。"""

    client = LiveAgentClient()
    result = await client.query_road_events(road=road)
    return dumps(result, ensure_ascii=False)


@tool("live_service_query")
async def live_service_query(keyword: str) -> str:
    """查询服务区、充电桩和商业配套信息。"""

    client = LiveAgentClient()
    result = await client.query_services(keyword=keyword)
    return dumps(result, ensure_ascii=False)


@tool("live_network_overview_query")
async def live_network_overview_query(
    query: str,
    scope: str = "全路网",
    report_type: str = "ad_hoc",
) -> str:
    """查询整体路网概况和报表数据。"""

    client = LiveAgentClient()
    result = await client.query_network_overview(
        scope=scope,
        query=query,
        report_type=report_type,
    )
    return dumps(result, ensure_ascii=False)
