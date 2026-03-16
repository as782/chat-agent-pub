"""回答节点模块。
负责构建模型上下文、执行普通回答补全，并持久化本轮输出。
当前阶段不负责工具循环编排，该能力已拆到独立 tool_node。
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context_builder import ContextBuilder, message_entity_to_input_message
from app.agent.state import AgentState, ChatExecutionRequest, ChatTurnResult, PreparedContext
from app.clients.llm_client import LlmChatCompletionResult, LlmClient, LlmInputMessage
from app.core.exceptions import AppException
from app.memory.manager import MemoryManager
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall, ToolRegistry

RECENT_CONTEXT_WINDOW_SIZE = 8


class AnswerNode:
    """LangGraph 回答节点。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
        context_builder: ContextBuilder | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._llm_client = llm_client or LlmClient()
        self._tool_registry = tool_registry or ToolRegistry()
        self._context_builder = context_builder or ContextBuilder()
        self._memory_manager = memory_manager or MemoryManager(db_session)
        self._message_repository = MessageRepository(db_session)

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行普通回答节点主逻辑。"""

        prepared_context_state = await self.prepare_context_state(state)
        prepared_context = prepared_context_state["prepared_context"]
        execution_request = self.build_execution_request_from_state(state)
        completion_result = await self._llm_client.create_chat_completion(
            messages=prepared_context.messages,
            model_name=execution_request.model_name,
        )
        final_result = await self.persist_completion_result(
            session_id=str(state["session_id"]),
            completion_result=completion_result,
            executed_tool_calls=[],
            used_session_memory=prepared_context.used_session_memory,
        )
        return {
            **prepared_context_state,
            "final_result": final_result,
        }

    async def prepare_context(
        self,
        execution_request: ChatExecutionRequest,
        *,
        knowledge_context: str | None = None,
        mcp_context: str | None = None,
    ) -> PreparedContext:
        """为流式和非流式路径统一准备模型上下文。"""

        if execution_request.session_id is None:
            raise AppException(
                "执行回答节点前必须先解析会话标识。",
                error_code="invalid_request",
            )

        recent_messages: list[LlmInputMessage] = []
        memory_summary: str | None = None
        if execution_request.need_session_memory:
            memory_snapshot = await self._memory_manager.load_snapshot(execution_request.session_id)
            recent_messages = await self._load_recent_messages(execution_request.session_id)
            memory_summary = memory_snapshot.summary

        return self._context_builder.build_context(
            input_messages=execution_request.input_messages,
            recent_messages=recent_messages,
            memory_summary=memory_summary,
            need_session_memory=execution_request.need_session_memory,
            knowledge_context=knowledge_context,
            mcp_context=mcp_context,
        )

    async def prepare_context_state(self, state: AgentState) -> dict[str, PreparedContext]:
        """从图状态中准备当前轮次上下文。"""

        execution_request = self._build_execution_request_from_state(state)
        knowledge_context = (
            str(state["knowledge_context"])
            if isinstance(state.get("knowledge_context"), str)
            else None
        )
        mcp_context = (
            str(state["mcp_context"]) if isinstance(state.get("mcp_context"), str) else None
        )
        prepared_context = await self.prepare_context(
            execution_request,
            knowledge_context=knowledge_context,
            mcp_context=mcp_context,
        )
        return {"prepared_context": prepared_context}

    async def persist_stream_result(
        self,
        *,
        session_id: str,
        completion_result: LlmChatCompletionResult,
        used_session_memory: bool,
    ) -> ChatTurnResult:
        """持久化流式路径最终得到的模型输出。"""

        return await self._persist_completion_result(
            session_id=session_id,
            completion_result=completion_result,
            executed_tool_calls=[],
            used_session_memory=used_session_memory,
        )

    async def persist_completion_result(
        self,
        *,
        session_id: str,
        completion_result: LlmChatCompletionResult,
        executed_tool_calls: list[ExecutedToolCall],
        used_session_memory: bool,
    ) -> ChatTurnResult:
        """持久化最终回答，并构造统一回包结果。"""

        if completion_result.tool_calls:
            await self._persist_assistant_tool_calls(
                session_id=session_id,
                completion_result=completion_result,
            )
        else:
            await self._message_repository.create(
                message_id=self._generate_identifier(),
                session_id=session_id,
                role="assistant",
                content=completion_result.content,
                message_metadata={
                    "finish_reason": completion_result.finish_reason,
                    "model_name": completion_result.model_name,
                },
            )

        return ChatTurnResult(
            session_id=session_id,
            content=completion_result.content,
            model_name=completion_result.model_name,
            prompt_tokens=completion_result.prompt_tokens,
            completion_tokens=completion_result.completion_tokens,
            total_tokens=completion_result.total_tokens,
            finish_reason=completion_result.finish_reason,
            tool_calls=executed_tool_calls,
            used_session_memory=used_session_memory,
        )

    async def _load_recent_messages(self, session_id: str) -> list[LlmInputMessage]:
        """读取当前会话最近上下文窗口。"""

        total_message_count = await self._message_repository.count_by_session(session_id)
        query_limit = min(max(total_message_count, 1), RECENT_CONTEXT_WINDOW_SIZE)
        query_offset = max(total_message_count - query_limit, 0)
        recent_message_entities = await self._message_repository.list_by_session(
            session_id,
            limit=query_limit,
            offset=query_offset,
        )
        return [
            message_entity_to_input_message(message_entity)
            for message_entity in recent_message_entities
        ]

    async def persist_assistant_tool_calls(
        self,
        *,
        session_id: str,
        completion_result: LlmChatCompletionResult,
    ) -> None:
        """持久化模型返回的工具调用请求。"""

        await self._message_repository.create(
            message_id=self._generate_identifier(),
            session_id=session_id,
            role="assistant",
            content=completion_result.content,
            message_metadata={
                "finish_reason": completion_result.finish_reason,
                "model_name": completion_result.model_name,
                "tool_calls": [
                    {
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in completion_result.tool_calls
                ],
            },
        )

    @staticmethod
    def build_execution_request_from_state(state: AgentState) -> ChatExecutionRequest:
        """把图状态恢复为统一执行请求。"""

        return ChatExecutionRequest(
            session_id=state.get("session_id"),
            need_session_memory=bool(state.get("need_session_memory", False)),
            latest_user_message=str(state.get("latest_user_message", "")),
            input_messages=list(state.get("input_messages", [])),
            model_name=state.get("model_name"),
            requested_tool_names=state.get("requested_tool_names"),
            tool_choice=state.get("tool_choice"),
            user_id=state.get("user_id"),
        )

    @staticmethod
    def _build_execution_request_from_state(state: AgentState) -> ChatExecutionRequest:
        """兼容旧调用入口。"""

        return AnswerNode.build_execution_request_from_state(state)

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
