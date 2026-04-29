"""路线起终点参数校验。

guard 是调用 /agent/driving 前的最后一道防线：允许未知地点，但不允许
“我现在正在...”或“杭州去”这类还没清洗干净的片段进入上游接口。
"""

from __future__ import annotations

from app.agent.od.normalizer import clean_endpoint
from app.agent.od.patterns import DIRTY_ENDPOINT_PATTERN, ROUTE_ACTION_SUFFIX_PATTERN


def is_valid_endpoint(value: object | None) -> bool:
    """判断单个端点是否已经干净到可以传给路线接口。"""

    raw_text = str(value or "").strip()
    text = clean_endpoint(value)
    # 如果 clean_endpoint 会改变原值，说明调用方传入的是未清洗片段，应先修复再调用。
    if raw_text != text:
        return False
    if len(text) < 2 or len(text) > 30:
        return False
    if DIRTY_ENDPOINT_PATTERN.search(text) is not None:
        return False
    # 端点不能以方向/动作词结尾，例如“杭州去”“宁波往”。
    if ROUTE_ACTION_SUFFIX_PATTERN.search(text) is not None:
        return False
    return True


def validate_route_arguments(
    origin: object | None,
    destination: object | None,
) -> tuple[bool, list[str]]:
    """校验完整路线参数。

    返回值第一项表示是否可调用路线接口；第二项是机器可读告警，
    供日志、测试和后续澄清逻辑使用。
    """

    warnings: list[str] = []
    clean_origin = clean_endpoint(origin)
    clean_destination = clean_endpoint(destination)
    if not is_valid_endpoint(origin):
        warnings.append("invalid_origin")
    if not is_valid_endpoint(destination):
        warnings.append("invalid_destination")
    if clean_origin and clean_destination and clean_origin == clean_destination:
        warnings.append("same_origin_destination")
    return (not warnings, warnings)
