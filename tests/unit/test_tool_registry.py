"""工具注册表单元测试。"""

from __future__ import annotations

import pytest

from app.core.exceptions import AppException
from app.tools.registry import ToolRegistry


def test_tool_registry_lists_builtin_tools() -> None:
    """验证注册表会返回当前内置工具名称。"""

    registry = ToolRegistry()

    assert registry.list_tool_names() == [
        "calculator",
        "current_datetime",
        "live_driving_query",
        "live_network_overview_query",
        "live_road_event_query",
        "live_service_query",
    ]


@pytest.mark.asyncio
async def test_tool_registry_executes_calculator_tool() -> None:
    """验证注册表可以执行计算器工具。"""

    registry = ToolRegistry()
    execution_results = await registry.execute_tool_calls(
        [{"id": "call_1", "name": "calculator", "args": {"expression": "1+1"}}]
    )

    assert execution_results[0].tool_name == "calculator"
    assert execution_results[0].output == "2"


def test_tool_registry_rejects_unknown_tool() -> None:
    """验证注册表会拒绝未注册的工具。"""

    registry = ToolRegistry()

    with pytest.raises(AppException) as exception_info:
        registry.ensure_supported(["unknown_tool"])

    assert exception_info.value.error_code == "unsupported_tool"
