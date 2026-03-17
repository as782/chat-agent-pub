"""业务节点单元测试。"""

import pytest

from app.agent.nodes.report_node import ReportNode
from app.agent.nodes.traffic_node import TrafficNode
from app.agent.state import ResolvedArguments


@pytest.mark.asyncio
async def test_traffic_node_builds_business_context() -> None:
    """路况节点应把结构化参数整理为上下文文本。"""

    node = TrafficNode()

    result = await node.run(
        {
            "resolved_arguments": ResolvedArguments(
                category="traffic_status",
                arguments={"target": "杭金衢高速", "time_range": "current"},
            )
        }
    )

    assert result["traffic_context"] is not None
    assert "杭金衢高速" in result["traffic_context"]


@pytest.mark.asyncio
async def test_report_node_builds_business_context() -> None:
    """报表节点应把结构化参数整理为上下文文本。"""

    node = ReportNode()

    result = await node.run(
        {
            "resolved_arguments": ResolvedArguments(
                category="network_report",
                arguments={"scope": "全路网", "need_table": True},
            )
        }
    )

    assert result["report_context"] is not None
    assert "全路网" in result["report_context"]
