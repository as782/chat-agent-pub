"""回答节点模块。

负责构建模型上下文、执行普通回答补全，并持久化本轮输出。
当前阶段不负责工具循环编排，这部分能力由独立的 tool_node 承担。
"""

from __future__ import annotations

from json import dumps
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context_builder import ContextBuilder, message_entity_to_input_message
from app.agent.prompts import (
    GENERAL_ANSWER_PROMPT,
    NETWORK_REPORT_SUMMARY_PROMPT,
    POLICY_SUMMARY_PROMPT,
    ROUTE_SUMMARY_PROMPT,
    TRAFFIC_SUMMARY_PROMPT,
)
from app.agent.state import (
    AgentState,
    ChatExecutionRequest,
    ChatTurnResult,
    ExecutorResult,
    PreparedContext,
)
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
            enable_thinking=execution_request.enable_thinking,
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
        answer_instruction: str | None = None,
        step_results: dict[str, ExecutorResult] | None = None,
        knowledge_context: str | None = None,
        route_context: str | None = None,
        mcp_context: str | None = None,
        traffic_context: str | None = None,
        report_context: str | None = None,
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
            answer_instruction=answer_instruction,
            executor_results_context=self._build_executor_results_context(step_results or {}),
            knowledge_context=knowledge_context,
            route_context=route_context,
            mcp_context=mcp_context,
            traffic_context=traffic_context,
            report_context=report_context,
        )

    async def prepare_context_state(self, state: AgentState) -> dict[str, PreparedContext]:
        """从图状态中准备当前轮次上下文。"""

        execution_request = self._build_execution_request_from_state(state)
        step_results = state.get("step_results", {})
        knowledge_context = (
            str(state["knowledge_context"])
            if isinstance(state.get("knowledge_context"), str)
            else None
        )
        route_context = (
            str(state["route_context"]) if isinstance(state.get("route_context"), str) else None
        )
        mcp_context = (
            str(state["mcp_context"]) if isinstance(state.get("mcp_context"), str) else None
        )
        traffic_context = (
            str(state["traffic_context"]) if isinstance(state.get("traffic_context"), str) else None
        )
        report_context = (
            str(state["report_context"]) if isinstance(state.get("report_context"), str) else None
        )
        prepared_context = await self.prepare_context(
            execution_request,
            answer_instruction=self._resolve_answer_instruction(state),
            step_results=step_results,
            knowledge_context=knowledge_context,
            route_context=route_context,
            mcp_context=mcp_context,
            traffic_context=traffic_context,
            report_context=report_context,
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

        return await self.persist_completion_result(
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
            await self.persist_assistant_tool_calls(
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
            enable_thinking=state.get("enable_thinking"),
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

    @staticmethod
    def _resolve_answer_instruction(state: AgentState) -> str:
        """根据当前主分类选择最终回答提示词。"""

        primary_category = state.get("primary_category", "general")
        if primary_category == "policy":
            return POLICY_SUMMARY_PROMPT
        if primary_category == "route_planning":
            return ROUTE_SUMMARY_PROMPT
        if primary_category == "traffic_status":
            return TRAFFIC_SUMMARY_PROMPT
        if primary_category == "network_report":
            return NETWORK_REPORT_SUMMARY_PROMPT
        return GENERAL_ANSWER_PROMPT

    @staticmethod
    def _build_executor_results_context(
        step_results: dict[str, ExecutorResult] | object,
    ) -> str | None:
        """把统一 step_results 转成可注入回答节点的结构化上下文。"""

        if not isinstance(step_results, dict) or not step_results:
            return None

        context_lines = ["以下是当前执行节点返回的结构化结果，请优先依据这些结果组织回答："]
        for step_id in sorted(step_results):
            executor_result = step_results[step_id]
            if not isinstance(executor_result, ExecutorResult):
                continue

            context_lines.append(
                f"[{step_id}] executor={executor_result.executor} "
                f"success={executor_result.is_success}"
            )
            if executor_result.summary:
                context_lines.append(f"summary={executor_result.summary}")
            if executor_result.normalized_result:
                context_lines.append(
                    dumps(executor_result.normalized_result, ensure_ascii=False, indent=2)
                )

        return "\n".join(context_lines) if len(context_lines) > 1 else None
