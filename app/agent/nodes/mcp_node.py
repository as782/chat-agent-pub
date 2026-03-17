"""MCP 节点模块。
负责在命中 MCP 路由时加载远端 MCP 工具，并将其转换为当前轮次可用的运行时工具。
当前阶段优先支持最小可用的工具发现与调用，不负责复杂多服务规划。
"""

from __future__ import annotations

from app.agent.state import AgentState, ExecutorResult, merge_step_result, resolve_execution_step_id
from app.core.exceptions import AppException
from app.mcp.manager import McpManager


class McpNode:
    """LangGraph MCP 节点。"""

    def __init__(self, *, mcp_manager: McpManager | None = None) -> None:
        self._mcp_manager = mcp_manager or McpManager()

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行 MCP 节点主逻辑。"""

        runtime_tools = await self._mcp_manager.build_runtime_tools()
        if not runtime_tools:
            raise AppException(
                "当前未发现可用的 MCP 工具。",
                error_code="mcp_no_available_tools",
            )

        step_id = resolve_execution_step_id(state, executor="mcp", default_step_id="mcp_1")
        executor_result = ExecutorResult(
            step_id=step_id,
            executor="mcp",
            is_success=True,
            raw_result={
                "tools": [
                    {
                        "registered_name": runtime_tool.registered_name,
                        "remote_tool_name": runtime_tool.remote_tool_name,
                        "server_name": runtime_tool.server_name,
                    }
                    for runtime_tool in runtime_tools
                ]
            },
            normalized_result={
                "tool_count": len(runtime_tools),
                "tool_names": [runtime_tool.registered_name for runtime_tool in runtime_tools],
            },
            summary=f"已发现 {len(runtime_tools)} 个可调用的 MCP 工具。",
            sources=list({runtime_tool.server_name for runtime_tool in runtime_tools}),
        )
        return {
            "mcp_context": self._mcp_manager.build_agent_context(runtime_tools),
            "mcp_tools": runtime_tools,
            **merge_step_result(state, result=executor_result),
        }
