"""Deterministic network report renderer tests."""

import logging

from app.agent.network_report_renderer import (
    build_network_report_render_result,
    render_network_report_from_step_results,
)
from app.agent.state import ExecutorResult


def test_network_report_renderer_filters_accident_and_vehicle_fault_events() -> None:
    """测试网络报告渲染器正确过滤交通事故(04)、车辆故障(07)事件和10110硬路肩开放管制。"""
    report_markdown = render_network_report_from_step_results(
        {
            "report_1": ExecutorResult(
                step_id="report_1",
                executor="report",
                is_success=True,
                normalized_result={
                    "congestion_total_mile": 5.0,
                    "congestion_top_items": [
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
                    "control_top_items": [
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
                    "exit_top_items": [
                        # 收费站管制 - 不应该被过滤（收费站管制不受事件类型影响）
                        {
                            "roadGBCode": "G60",
                            "roadName": "沪昆高速",
                            "directionName": "杭州方向",
                            "tollName": "嘉兴收费站",
                            "entrance": 1,
                            "controlTypeName": "关闭",
                        }
                    ],
                },
            )
        }
    )

    assert report_markdown is not None
    
    # 验证交通事故、车辆故障和硬路肩开放事件没有出现在结果中
    # G2 (京沪高速)、G42 (沪蓉高速) 和 G50 (沪渝高速) 应该完全不在结果中，因为它们只有被过滤的事件
    assert "| G2 |" not in report_markdown
    assert "| G97 |" not in report_markdown
    assert "| G42 |" not in report_markdown
    assert "| G50 |" not in report_markdown
    
    # 验证正常事件仍然存在
    assert "| G15 |" in report_markdown  # 沈海高速 - 正常拥堵
    assert "| G56 |" in report_markdown  # 杭瑞高速 - 正常施工
    assert "| G60 |" in report_markdown  # 沪昆高速 - 有收费站管制（未被过滤）
    
    # 验证具体的描述内容
    assert "车流量大，缓行3公里" in report_markdown
    assert "封闭部分车道" in report_markdown  # 控制类型字段
    assert "道路施工" in report_markdown     # 事件分类字段
    assert "嘉兴收费站" in report_markdown   # 收费站管制


def test_network_report_renderer_splits_exit_and_mainline_rows_stably() -> None:
    report_markdown = render_network_report_from_step_results(
        {
            "report_1": ExecutorResult(
                step_id="report_1",
                executor="report",
                is_success=True,
                normalized_result={
                    "congestion_total_mile": 0,
                    "congestion_top_items": [],
                    "control_top_items": [
                        {
                            "roadGBCode": "G1512",
                            "roadName": "G1512甬金高速（金华段）",
                            "directionName": "宁波方向",
                            "des": "宁波方向，单向封道",
                        }
                    ],
                    "exit_top_items": [
                        {
                            "roadGBCode": "G1512",
                            "roadName": "G1512甬金高速（金华段）",
                            "directionName": "宁波方向",
                            "tollName": "佛堂收费站",
                            "entrance": 1,
                            "controlTypeName": "关闭",
                        },
                        {
                            "roadGBCode": "G1512",
                            "roadName": "G1512甬金高速（金华段）",
                            "directionName": "宁波方向",
                            "tollName": "佛堂收费站",
                            "entrance": 0,
                            "controlTypeName": "分流",
                        },
                    ],
                },
            )
        }
    )

    assert report_markdown is not None
    assert "| G1512 |" in report_markdown
    assert "佛堂收费站" in report_markdown
    assert "宁波方向" in report_markdown


def test_network_report_renderer_accepts_serialized_step_results() -> None:
    render_result = build_network_report_render_result(
        {
            "report_1": {
                "step_id": "report_1",
                "executor": "report",
                "is_success": True,
                "normalized_result": {
                    "congestion_total_mile": 3.0,
                    "congestion_top_items": [],
                    "control_top_items": [],
                    "exit_top_items": [
                        {
                            "roadGBCode": "G1512",
                            "roadName": "G1512甬金高速（金华段）",
                            "directionName": "宁波方向",
                            "tollName": "佛堂收费站",
                            "entrance": 1,
                            "controlTypeName": "关闭",
                        }
                    ],
                },
            }
        }
    )

    assert render_result is not None
    assert "| G1512 |" in render_result.to_markdown()


def test_network_report_renderer_logs_generated_table(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="app.agent.network_report_renderer"):
        render_result = build_network_report_render_result(
            {
                "report_1": ExecutorResult(
                    step_id="report_1",
                    executor="report",
                    is_success=True,
                    normalized_result={
                        "congestion_total_mile": 0,
                        "congestion_top_items": [],
                        "control_top_items": [],
                        "exit_top_items": [
                            {
                                "roadGBCode": "G60",
                                "roadName": "G60沪昆高速",
                                "directionName": "杭州方向",
                                "tollName": "嘉兴收费站",
                                "entrance": 1,
                                "controlTypeName": "关闭",
                            }
                        ],
                    },
                )
            }
        )

    assert render_result is not None
    assert "Network report table generated" in caplog.text
    assert "| roadCode | highwayName | roadSection | controls | traffic |" in caplog.text
    assert "| G60 |" in caplog.text
