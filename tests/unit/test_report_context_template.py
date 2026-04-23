"""Network report context template tests."""

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
                "code": 0,
                "data": {
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
                            "eventClass": "03",
                            "eventType": "105",
                            "subEventType": "施工缓行",
                            "des": "金华方向缓行",
                        }
                    ],
                    "controlTopN": [
                        {
                            "roadGBCode": "G25",
                            "roadName": "长深高速",
                            "direction": "0",
                            "tollName": "诸暨北收费站",
                            "entrance": 1,
                            "controlTypeName": "入口管制",
                            "limitMeasureTypeName": "货车分流",
                            "startTime": "2026-03-31 08:10:00",
                            "endTime": "2026-03-31 12:00:00",
                            "des": "入口限流",
                        }
                    ],
                    "exitTopN": [
                        {
                            "roadGbCode": "G60",
                            "roadName": "沪昆高速",
                            "directionName": "下行",
                            "tollName": "萧山收费站",
                            "entrance": 0,
                            "controlTypeName": "关闭",
                            "limitMeasureTypeName": "收费站管制",
                            "startTime": "2026-03-31 09:30:00",
                            "endTime": "2026-03-31 11:30:00",
                            "des": "收费站关闭",
                        }
                    ],
                },
                "message": "",
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
                arguments={"query": "生成全路网路况对比表格"},
            ),
        }
    )

    report_context = result["report_context"]
    assert report_context is not None
    assert "查询时间：2026-03-31 09:00:00" in report_context
    assert "拥堵总里程：12.5 公里" in report_context
    assert "拥堵汇总（1条）：" in report_context
    assert "G60 / 沪昆高速" in report_context
    assert "方向 上行" in report_context
    assert "区间 K120-K128" in report_context
    assert "事件分类 交通气象（道路缓行）" in report_context
    assert "施工缓行" not in report_context
    assert "主线管制（1条）：" in report_context
    assert "G25 / 长深高速" in report_context
    assert "收费站 诸暨北收费站" in report_context
    assert "收费站管制（1条）：" in report_context
    assert "收费站 萧山收费站" in report_context
    assert "api_result" not in report_context
    assert "queryTime" not in report_context

    step_results = result["step_results"]
    assert "report_1" in step_results
    normalized_result = step_results["report_1"].normalized_result
    assert normalized_result["query_time"] == "2026-03-31 09:00:00"
    assert normalized_result["congestion_top_count"] == 1
    assert normalized_result["control_top_count"] == 1
    assert normalized_result["exit_top_count"] == 1
