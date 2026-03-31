"""直播问答工具定义模块。

负责把直播问答接口封装为标准工具定义，供统一工具注册表按需挂载。
当前阶段保持工具名稳定，便于业务节点通过名称调用。
"""

from __future__ import annotations

from json import dumps

from langchain_core.tools import BaseTool, tool

from app.tools.live_agent.client import LiveAgentClient


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
async def live_network_overview_query() -> str:
    """查询整体路网概况和报表数据。"""

    client = LiveAgentClient()
    result = await client.query_network_overview()
    return dumps(result, ensure_ascii=False)


def build_live_agent_tools() -> dict[str, BaseTool]:
    """返回直播问答工具集注册映射。"""

    return {
        "live_driving_query": live_driving_query,
        "live_road_event_query": live_road_event_query,
        "live_service_query": live_service_query,
        "live_network_overview_query": live_network_overview_query,
    }
