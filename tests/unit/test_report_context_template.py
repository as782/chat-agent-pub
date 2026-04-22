"""路网报告上下文模板测试。"""

from json import dumps

import pytest

from app.agent.nodes.report_node import ReportNode
from app.agent.state import ExecutionPlan, ResolvedArguments


class _TemplateReportToolRegistry:
    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name != "live_network_overview_query":
            raise AssertionError(f"unexpected tool: {tool_name}")
        del arguments
        return dumps(
            {
                "queryTime": "2026-03-31 09:00:00",
                "congestion": {"totalMile": 12.5},
                "congestionTopN": [
                    {
                        "roadGBCode": "G60",
                        "roadName": "沪昆高速",
                        "directionType": "1",
                        "beginMilestone": 120,
                        "endMilestone": 128,
                        "roadAmbleMile": 8.0,
                        "controlMeasures": "借道通行",
                        "situationRemark": "主线缓行",
                        "jeeves": "占用1车道",
                        "beginTime": "2026-03-31 08:20:00",
                        "expectedTime": "2026-03-31 10:00:00",
                        "des": "金华方向缓行",
                    }
                ],
                "accidentTopN": [
                    {
                        "roadGBCode": "G2501",
                        "roadName": "杭州绕城高速",
                        "directionType": "2",
                        "beginMilestone": 32,
                        "endMilestone": 36,
                        "roadAmbleMile": 4.5,
                        "controlMeasures": "封闭应急车道",
                        "situationRemark": "事故处理",
                        "jeeves": "占用2车道",
                        "beginTime": "2026-03-31 08:40:00",
                        "expectedTime": "2026-03-31 11:30:00",
                        "des": "追尾事故",
                    }
                ],
                "controlTopN": [
                    {
                        "roadGBCode": "G25",
                        "roadName": "长深高速",
                        "directionType": "0",
                        "beginMilestone": 210,
                        "endMilestone": 212,
                        "roadAmbleMile": 0.0,
                        "controlMeasures": "货车分流",
                        "situationRemark": "入口管制",
                        "jeeves": "占用0车道",
                        "beginTime": "2026-03-31 08:10:00",
                        "expectedTime": "2026-03-31 12:00:00",
                        "des": "入口限流",
                    }
                ],
            },
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_report_node_builds_template_context() -> None:
    node = ReportNode(tool_registry=_TemplateReportToolRegistry())

    result = await node.run(
        {
            "execution_plan": ExecutionPlan(
                primary_category="network_report",
                execution_mode="single_step",
                recommended_route="report",
            ),
            "resolved_arguments": ResolvedArguments(
                category="network_report",
                arguments={"query": "请生成今日全路网路况对比表格"},
            ),
        }
    )

    report_context = result["report_context"]
    assert report_context is not None
    assert "查询时间：2026-03-31 09:00:00" in report_context
    assert "拥堵总里程：12.5 公里" in report_context
    assert "拥堵列表：" in report_context
    assert "- G60，沪昆高速，方向：上行，K120~K128，缓行约 8.0 公里，管制措施：借道通行，现场情况备注：主线缓行，占道情况：占用1车道，开始时间：2026-03-31 08:20:00-预期结束时间：2026-03-31 10:00:00，事件描述：金华方向缓行" in report_context
    assert "事故列表：" in report_context
    assert "- G2501，杭州绕城高速，方向：下行，K32~K36，缓行约 4.5 公里，管制措施：封闭应急车道，现场情况备注：事故处理，占道情况：占用2车道，开始时间：2026-03-31 08:40:00-预期结束时间：2026-03-31 11:30:00，事件描述：追尾事故" in report_context
    assert "管制列表：" in report_context
    assert "- G25，长深高速，方向：双向，K210~K212，缓行约 0.0 公里，管制措施：货车分流，现场情况备注：入口管制，占道情况：占用0车道，开始时间：2026-03-31 08:10:00-预期结束时间：2026-03-31 12:00:00，事件描述：入口限流" in report_context
    assert "api_result" not in report_context
    assert "queryTime" not in report_context
