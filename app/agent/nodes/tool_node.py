"""工具节点模块。
负责在命中工具或 MCP 路由时执行模型补全、工具调用和二次补全循环。
当前阶段支持内置工具与 MCP 远端工具混合执行，不负责更复杂的跨服务规划。
"""

from __future__ import annotations

from json import dumps
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.nodes.answer_node import AnswerNode
from app.agent.state import AgentState
from app.clients.llm_client import (
    LlmBindableTool,
    LlmChatCompletionResult,
    LlmClient,
    LlmInputMessage,
)
from app.core.exceptions import AppException
from app.mcp.manager import McpManager
from app.mcp.models import McpRuntimeTool
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall, ToolRegistry

MAX_TOOL_CALL_ROUNDS = 10


class ToolNode:
    """LangGraph 工具节点。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        answer_node: AnswerNode,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_manager: McpManager | None = None,
    ) -> None:
        self._answer_node = answer_node
        self._llm_client = llm_client or LlmClient()
        self._tool_registry = tool_registry or ToolRegistry()
        self._mcp_manager = mcp_manager or McpManager()
        self._message_repository = MessageRepository(db_session)

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行工具节点主逻辑。"""

        prepared_context_state = await self._answer_node.prepare_context_state(state)
        prepared_context = prepared_context_state["prepared_context"]
        execution_request = self._answer_node.build_execution_request_from_state(state)
        runtime_mcp_tools = self._extract_runtime_mcp_tools(state)
        available_tools = self._build_available_tools(
            route=str(state.get("route", "answer")),
            requested_tool_names=execution_request.requested_tool_names,
            runtime_mcp_tools=runtime_mcp_tools,
        )
        completion_result, executed_tool_calls = await self._execute_tool_completion_loop(
            messages=prepared_context.messages,
            model_name=execution_request.model_name,
            session_id=str(state["session_id"]),
            available_tools=available_tools,
            tool_choice=execution_request.tool_choice,
            enable_thinking=execution_request.enable_thinking,
            runtime_mcp_tools=runtime_mcp_tools,
        )
        final_result = await self._answer_node.persist_completion_result(
            session_id=str(state["session_id"]),
            completion_result=completion_result,
            executed_tool_calls=executed_tool_calls,
            used_session_memory=prepared_context.used_session_memory,
        )
        return {
            **prepared_context_state,
            "final_result": final_result,
        }

    def build_available_tools(
        self,
        *,
        route: str,
        requested_tool_names: list[str] | None,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[LlmBindableTool]:
        """构造当前轮次可绑定给模型的工具集合。"""

        return self._build_available_tools(
            route=route,
            requested_tool_names=requested_tool_names,
            runtime_mcp_tools=runtime_mcp_tools,
        )

    async def execute_requested_tools_and_persist(
        self,
        *,
        session_id: str,
        completion_result: LlmChatCompletionResult,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[ExecutedToolCall]:
        """执行模型请求的工具，并将 tool 消息持久化。"""

        executed_tool_calls = await self._execute_requested_tools(
            completion_result=completion_result,
            runtime_mcp_tools=runtime_mcp_tools,
        )
        await self._persist_tool_messages(
            session_id=session_id,
            executed_tool_calls=executed_tool_calls,
        )
        return executed_tool_calls

    async def _execute_tool_completion_loop(
        self,
        *,
        messages: list[LlmInputMessage],
        model_name: str | None,
        session_id: str,
        available_tools: list[LlmBindableTool],
        tool_choice: str | dict[str, object] | None,
        enable_thinking: bool | None,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> tuple[LlmChatCompletionResult, list[ExecutedToolCall]]:
        """执行带工具的模型补全循环。"""

        conversation_messages = list(messages)
        normalized_tool_choice = tool_choice or "auto"
        executed_tool_calls: list[ExecutedToolCall] = []

        for tool_round in range(MAX_TOOL_CALL_ROUNDS):
            completion_result = await self._llm_client.create_chat_completion(
                messages=conversation_messages,
                model_name=model_name,
                tools=available_tools,
                tool_choice=normalized_tool_choice,
                enable_thinking=enable_thinking,
            )
            if not completion_result.tool_calls:
                return completion_result, executed_tool_calls

            await self._answer_node.persist_assistant_tool_calls(
                session_id=session_id,
                completion_result=completion_result,
            )
            conversation_messages.append(
                LlmInputMessage(
                    role="assistant",
                    content=completion_result.content,
                    tool_calls=completion_result.tool_calls,
                )
            )
            current_tool_results = await self._execute_requested_tools(
                completion_result=completion_result,
                runtime_mcp_tools=runtime_mcp_tools,
            )
            await self._persist_tool_messages(
                session_id=session_id,
                executed_tool_calls=current_tool_results,
            )
            for tool_result in current_tool_results:
                executed_tool_calls.append(tool_result)
                conversation_messages.append(
                    LlmInputMessage(
                        role="tool",
                        content=tool_result.output,
                        tool_call_id=tool_result.tool_call_id,
                    )
                )
            normalized_tool_choice = "auto"
            if tool_round + 1 >= MAX_TOOL_CALL_ROUNDS:
                raise AppException(
                    "工具调用轮次超过当前上限。",
                    error_code="tool_call_limit_exceeded",
                )

        raise AppException(
            "模型未返回最终回答。",
            error_code="invalid_llm_response",
        )

    def _build_available_tools(
        self,
        *,
        route: str,
        requested_tool_names: list[str] | None,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[LlmBindableTool]:
        """根据路由和状态构造当前轮次允许使用的工具集合。"""

        builtin_tools = (
            []
            if route == "mcp"
            else self._tool_registry.get_tools(requested_tool_names)
            if requested_tool_names is not None
            else []
        )
        mcp_tools = [runtime_tool.to_openai_tool() for runtime_tool in runtime_mcp_tools]
        available_tools = [*builtin_tools, *mcp_tools]
        if not available_tools:
            raise AppException(
                "当前请求没有可执行的工具。",
                error_code="no_available_tools",
            )
        return available_tools

    async def _execute_requested_tools(
        self,
        *,
        completion_result: LlmChatCompletionResult,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[ExecutedToolCall]:
        """执行模型本轮返回的全部工具调用。"""

        mcp_tool_map = {
            runtime_tool.registered_name: runtime_tool for runtime_tool in runtime_mcp_tools
        }
        builtin_tool_calls: list[dict[str, object]] = []
        executed_tool_calls: list[ExecutedToolCall] = []

        for tool_call in completion_result.tool_calls:
            runtime_mcp_tool = mcp_tool_map.get(tool_call.tool_name)
            if runtime_mcp_tool is None:
                builtin_tool_calls.append(
                    {
                        "id": tool_call.tool_call_id,
                        "name": tool_call.tool_name,
                        "args": tool_call.arguments,
                    }
                )
                continue

            tool_response = await self._mcp_manager.call_tool(
                server_name=runtime_mcp_tool.server_name,
                tool_name=runtime_mcp_tool.remote_tool_name,
                arguments=tool_call.arguments,
            )
            normalized_output = tool_response.output_text
            if not normalized_output and tool_response.structured_content is not None:
                normalized_output = dumps(tool_response.structured_content, ensure_ascii=False)
            if not normalized_output:
                normalized_output = "MCP 工具未返回文本结果。"
            executed_tool_calls.append(
                ExecutedToolCall(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=runtime_mcp_tool.registered_name,
                    arguments=tool_call.arguments,
                    output=normalized_output,
                )
            )

        if builtin_tool_calls:
            executed_tool_calls.extend(
                await self._tool_registry.execute_tool_calls(builtin_tool_calls)
            )

        return executed_tool_calls

    async def _persist_tool_messages(
        self,
        *,
        session_id: str,
        executed_tool_calls: list[ExecutedToolCall],
    ) -> None:
        """把工具执行结果写入消息历史。"""

        for tool_result in executed_tool_calls:
            await self._message_repository.create(
                message_id=uuid4().hex,
                session_id=session_id,
                role="tool",
                content=tool_result.output,
                message_metadata={
                    "tool_call_id": tool_result.tool_call_id,
                    "tool_name": tool_result.tool_name,
                    "arguments": tool_result.arguments,
                },
            )

    @staticmethod
    def extract_runtime_mcp_tools(state: AgentState) -> list[McpRuntimeTool]:
        """从状态中提取当前轮次的 MCP 运行时工具。"""

        raw_runtime_mcp_tools = state.get("mcp_tools", [])
        if not isinstance(raw_runtime_mcp_tools, list):
            return []
        return [
            runtime_tool
            for runtime_tool in raw_runtime_mcp_tools
            if isinstance(runtime_tool, McpRuntimeTool)
        ]

    @staticmethod
    def _extract_runtime_mcp_tools(state: AgentState) -> list[McpRuntimeTool]:
        """兼容旧调用入口。"""

        return ToolNode.extract_runtime_mcp_tools(state)
