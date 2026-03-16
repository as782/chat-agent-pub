"""工具节点模块。

负责在命中工具路由时执行模型补全、工具调用和二次补全循环。
当前阶段只支持内置工具，不负责跨进程远程工具编排。
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.nodes.answer_node import AnswerNode
from app.agent.state import AgentState
from app.clients.llm_client import LlmChatCompletionResult, LlmClient, LlmInputMessage
from app.core.exceptions import AppException
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall, ToolRegistry

MAX_TOOL_CALL_ROUNDS = 3


class ToolNode:
    """LangGraph 工具节点。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        answer_node: AnswerNode,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._answer_node = answer_node
        self._llm_client = llm_client or LlmClient()
        self._tool_registry = tool_registry or ToolRegistry()
        self._message_repository = MessageRepository(db_session)

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行工具节点主逻辑。"""

        prepared_context_state = await self._answer_node.prepare_context_state(state)
        prepared_context = prepared_context_state["prepared_context"]
        execution_request = self._answer_node.build_execution_request_from_state(state)
        completion_result, executed_tool_calls = await self._execute_tool_completion_loop(
            messages=prepared_context.messages,
            model_name=execution_request.model_name,
            session_id=str(state["session_id"]),
            requested_tool_names=execution_request.requested_tool_names,
            tool_choice=execution_request.tool_choice,
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

    async def _execute_tool_completion_loop(
        self,
        *,
        messages: list[LlmInputMessage],
        model_name: str | None,
        session_id: str,
        requested_tool_names: list[str] | None,
        tool_choice: str | dict[str, object] | None,
    ) -> tuple[LlmChatCompletionResult, list[ExecutedToolCall]]:
        """执行带工具的模型补全循环。"""

        available_tools = self._tool_registry.get_tools(requested_tool_names)
        conversation_messages = list(messages)
        normalized_tool_choice = tool_choice or "auto"
        executed_tool_calls: list[ExecutedToolCall] = []

        for tool_round in range(MAX_TOOL_CALL_ROUNDS):
            completion_result = await self._llm_client.create_chat_completion(
                messages=conversation_messages,
                model_name=model_name,
                tools=available_tools,
                tool_choice=normalized_tool_choice,
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
            current_tool_results = await self._tool_registry.execute_tool_calls(
                [
                    {
                        "id": tool_call.tool_call_id,
                        "name": tool_call.tool_name,
                        "args": tool_call.arguments,
                    }
                    for tool_call in completion_result.tool_calls
                ]
            )
            for tool_result in current_tool_results:
                executed_tool_calls.append(tool_result)
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
