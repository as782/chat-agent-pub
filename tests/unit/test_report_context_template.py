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
                            "beginMilestoneStr": "K120+0",
                            "endMilestone": 128,
                            "endMilestoneStr": "K128+0",
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
                    "majorTopN": [
                        {
                            "roadGBCode": "G92",
                            "roadName": "杭州湾环线高速",
                            "directionName": "宁波方向",
                            "beginMilestone": 20,
                            "endMilestone": 22,
                            "beginTime": "2026-03-31 09:10:00",
                            "expectedTime": "2026-03-31 11:00:00",
                            "eventClass": "07",
                            "eventType": "07",
                            "subEventTypeId": "070701",
                            "des": "大流量通行缓慢",
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


class _FilteredReportToolRegistry:
    async def execute_named_tool(self, *, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name == "live_network_overview_query":
            return dumps(
                {
                    "data": {
                        "queryTime": "2026-04-24 21:30:00",
                        "congestionTotalMile": 8.5,
                        "congestionTopN": [
                            # 交通事故事件 - 应该被过滤
                            {
                                "roadGBCode": "G60",
                                "roadName": "沪昆高速",
                                "directionName": "上海方向",
                                "eventClass": "04",  # 交通事故
                                "des": "发生交通事故，多车追尾",
                            },
                            # 车辆故障事件 - 应该被过滤  
                            {
                                "roadGBCode": "G2",
                                "roadName": "京沪高速",
                                "directionName": "北京方向",
                                "eventClass": "07",  # 车辆故障
                                "des": "货车抛锚占用应急车道",
                            },
                            {
                                "roadGBCode": "G97",
                                "roadName": "测试高速",
                                "directionName": "北向",
                                "eventType": "97",
                                "des": "eventType车辆故障",
                            },
                            # 正常拥堵事件 - 应该保留
                            {
                                "roadGBCode": "G15",
                                "eventType": "07",
                                "roadName": "沈海高速",
                                "directionName": "广州方向",
                                "eventClass": "03",  # 道路缓行
                                "des": "车流量大，缓行3公里",
                            }
                        ],
                        "controlTopN": [
                            # 交通事故管制 - 应该被过滤
                            {
                                "roadGBCode": "G42",
                                "roadName": "沪蓉高速",
                                "directionName": "成都方向",
                                "eventClass": "04",  # 交通事故
                                "controlTypeName": "单向封道",
                                "des": "因交通事故实施单向封道",
                            },
                            # 硬路肩开放管制 - 应该被过滤
                            {
                                "roadGBCode": "G50",
                                "roadName": "沪渝高速",
                                "directionName": "重庆方向",
                                "controlType": "10110",  # 硬路肩开放
                                "controlTypeName": "硬路肩开放",
                                "des": "开放硬路肩供车辆通行",
                            },
                            # 正常施工管制 - 应该保留
                            {
                                "roadGBCode": "G56",
                                "roadName": "杭瑞高速",
                                "directionName": "杭州方向",
                                "eventClass": "05",  # 路面施工
                                "controlTypeName": "封闭部分车道",
                                "des": "道路施工，封闭右侧车道",
                            }
                        ],
                        "exitTopN": [
                            # 收费站管制 - 不应该被过滤
                            {
                                "roadGBCode": "G60",
                                "roadName": "沪昆高速",
                                "directionName": "杭州方向",
                                "tollName": "嘉兴收费站",
                                "entrance": 1,
                                "controlTypeName": "关闭",
                            }
                        ],
                    }
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"Unexpected tool call: {tool_name}")


@pytest.mark.asyncio
async def test_report_node_filters_accident_and_vehicle_fault_events() -> None:
    """测试report_node正确过滤交通事故(04)、车辆故障(07)事件和10110硬路肩开放管制。"""
    node = ReportNode(tool_registry=_FilteredReportToolRegistry())

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
    
    # 验证交通事故、车辆故障和硬路肩开放事件没有出现在report_context中
    assert "多车追尾" not in report_context
    assert "货车抛锚" not in report_context
    assert "eventType车辆故障" not in report_context
    assert "因交通事故实施单向封道" not in report_context
    assert "开放硬路肩供车辆通行" not in report_context
    
    # 验证正常事件仍然存在
    assert "车流量大，缓行3公里" in report_context
    assert "道路施工，封闭右侧车道" in report_context
    assert "嘉兴收费站" in report_context
    
    # 验证高速公路代码
    assert "G15" in report_context  # 沈海高速 - 正常拥堵
    assert "G56" in report_context  # 杭瑞高速 - 正常施工  
    assert "G60" in report_context  # 沪昆高速 - 有收费站管制
    
    step_results = result["step_results"]
    assert "report_1" in step_results
    normalized_result = step_results["report_1"].normalized_result
    
    # 验证normalized_result中的数据也被正确过滤
    congestion_items = normalized_result["congestion_top_items"]
    control_items = normalized_result["control_top_items"]
    exit_items = normalized_result["exit_top_items"]
    
    # 拥堵事件应该只有1个（过滤掉了2个）
    assert len(congestion_items) == 1
    assert congestion_items[0]["roadGBCode"] == "G15"
    
    # 主线管制应该只有1个（过滤掉了2个：交通事故和硬路肩开放）
    assert len(control_items) == 1
    assert control_items[0]["roadGBCode"] == "G56"
    
    # 收费站管制应该有1个（未被过滤）
    assert len(exit_items) == 1
    assert exit_items[0]["roadGBCode"] == "G60"


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
    assert "区间 K120+0-K128+0" in report_context
    assert "事件分类 交通气象（道路缓行）" in report_context
    assert "施工缓行" not in report_context
    assert "重大事件（1条）：" in report_context
    assert "G92 / 杭州湾环线高速" in report_context
    assert "事件分类 重大事件 / 小类 大流量" in report_context
    assert "大流量通行缓慢" in report_context
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
    assert normalized_result["major_top_count"] == 1
    assert normalized_result["control_top_count"] == 1
    assert normalized_result["exit_top_count"] == 1
