"""Deterministic network report renderer tests."""

from app.agent.network_report_renderer import (
    build_network_report_render_result,
    render_network_report_from_step_results,
)
from app.agent.state import ExecutorResult


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
