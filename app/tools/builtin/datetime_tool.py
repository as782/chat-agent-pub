"""时间工具模块。
负责提供当前时间查询能力，便于模型处理时间相关问题。
当前阶段只返回格式化后的当前时间，不负责日历计算和复杂时间推理。
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool("current_datetime")
def current_datetime_tool(timezone_name: str = "Asia/Shanghai") -> str:
    """返回指定时区的当前时间。"""

    normalized_timezone_name = timezone_name.strip() or "Asia/Shanghai"

    try:
        timezone = ZoneInfo(normalized_timezone_name)
    except ZoneInfoNotFoundError:
        timezone = UTC
        normalized_timezone_name = "UTC"

    current_time = datetime.now(timezone)
    return (
        f"当前时区: {normalized_timezone_name}\n"
        f"当前时间: {current_time.isoformat(timespec='seconds')}"
    )
