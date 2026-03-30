"""工具注册表模块。
负责管理内置工具的注册、筛选与执行，避免服务层直接依赖具体工具实现。
当前阶段只管理项目内置工具，不负责动态加载第三方工具与跨进程调度。
"""

from __future__ import annotations

from dataclasses import dataclass
from json import dumps
from typing import Any

from langchain_core.tools import BaseTool

from app.core.exceptions import AppException
from app.tools.builtin.calculator import calculator_tool
from app.tools.builtin.datetime_tool import current_datetime_tool
from app.tools.builtin.live_agent import (
    live_driving_query,
    live_network_overview_query,
    live_road_event_query,
    live_service_query,
)


@dataclass(slots=True)
class ExecutedToolCall:
    """已执行工具调用结果。"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    output: str


def tool_to_langchain_format(tool: BaseTool) -> BaseTool:
    """将工具转换为 LangChain 格式，供 langgraph.prebuilt.ToolNode 使用。"""
    return tool


class ToolRegistry:
    """内置工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {
            "calculator": calculator_tool,
            "current_datetime": current_datetime_tool,
            "live_driving_query": live_driving_query,
            "live_road_event_query": live_road_event_query,
            "live_service_query": live_service_query,
            "live_network_overview_query": live_network_overview_query,
        }

    def list_tool_names(self) -> list[str]:
        """返回当前可用的工具名称列表。"""

        return sorted(self._tools)

    def get_tools(self, tool_names: list[str] | None = None) -> list[BaseTool]:
        """按名称获取 LangChain 工具对象列表。"""

        if not tool_names:
            return list(self._tools.values())

        self.ensure_supported(tool_names)
        return [self._tools[tool_name] for tool_name in tool_names]

    def get_tool(self, tool_name: str) -> BaseTool:
        """按名称获取单个工具对象。"""

        self.ensure_supported([tool_name])
        return self._tools[tool_name]

    def ensure_supported(self, tool_names: list[str]) -> None:
        """校验请求中的工具名是否都已注册。"""

        unsupported_tool_names = [
            tool_name for tool_name in tool_names if tool_name not in self._tools
        ]
        if unsupported_tool_names:
            raise AppException(
                "请求中包含当前系统不支持的工具。",
                error_code="unsupported_tool",
                details={"tool_names": unsupported_tool_names},
            )

    def normalize_tool_choice(
        self, tool_choice: str | dict[str, Any] | None
    ) -> str | dict[str, Any] | None:
        """规范化工具选择参数。"""

        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "none", "required", "any"}:
                return tool_choice
            self.ensure_supported([tool_choice])
            return {
                "type": "function",
                "function": {"name": tool_choice},
            }

        if isinstance(tool_choice, dict):
            function_payload = tool_choice.get("function")
            function_name = (
                function_payload.get("name") if isinstance(function_payload, dict) else None
            )
            if isinstance(function_name, str):
                self.ensure_supported([function_name])
            return tool_choice

        raise AppException(
            "工具选择参数格式不正确。",
            error_code="invalid_request",
            details={"tool_choice": dumps(tool_choice, ensure_ascii=False, default=str)},
        )

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ExecutedToolCall]:
        """顺序执行工具调用并返回执行结果。"""

        executed_tool_calls: list[ExecutedToolCall] = []
        for tool_call in tool_calls:
            tool_name = str(tool_call["name"])
            self.ensure_supported([tool_name])

            tool_arguments = tool_call.get("args", {})
            if not isinstance(tool_arguments, dict):
                raise AppException(
                    "工具参数必须是 JSON 对象。",
                    error_code="invalid_tool_arguments",
                    details={"tool_name": tool_name},
                )

            tool_instance = self._tools[tool_name]
            tool_output = await tool_instance.ainvoke(tool_arguments)
            executed_tool_calls.append(
                ExecutedToolCall(
                    tool_call_id=str(tool_call["id"]),
                    tool_name=tool_name,
                    arguments=tool_arguments,
                    output=str(tool_output),
                )
            )

        return executed_tool_calls

    async def execute_named_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """按名称执行单个工具并返回字符串结果。"""

        self.ensure_supported([tool_name])
        tool_instance = self._tools[tool_name]
        tool_output = await tool_instance.ainvoke(arguments)
        return str(tool_output)
