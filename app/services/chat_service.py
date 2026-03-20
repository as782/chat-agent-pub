"""对话服务模块。
负责内部聊天接口的会话落库、LangGraph 对话编排与 OpenAI 兼容响应构建。
当前阶段负责把非流式和流式链路统一到同一套图状态准备逻辑中，不负责知识库和 MCP 的底层协议实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import AIMessageChunk
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import ConversationGraph
from app.agent.state import (
    ChatExecutionRequest,
    ChatTurnResult,
)
from app.clients.llm_client import LlmClient
from app.core.exceptions import AppException, ResourceNotFoundException
from app.core.logger import get_logger
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.openai_compat import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from app.services.openai_compat_service import OpenAICompatService
from app.tools.registry import ToolRegistry

LOGGER = get_logger(__name__)
_GRAPH_NODE_NAMES = {
    "planner_node",
    "argument_node",
    "scheduler_node",
    "router_node",
    "route_node",
    "tool_node",
    "ragflow_node",
    "mcp_node",
    "traffic_node",
    "report_node",
    "answer_node",
    "memory_node",
}
_USER_VISIBLE_STREAM_NODES = {"answer_node"}


class ChatService:
    """内部聊天服务。"""

    def __init__(
        self,
        db_session: AsyncSession,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._db_session = db_session
        self._session_repository = SessionRepository(db_session)
        self._message_repository = MessageRepository(db_session)
        self._llm_client = llm_client or LlmClient()
        self._tool_registry = tool_registry or ToolRegistry()
        self._conversation_graph = ConversationGraph(
            db_session,
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
        )
        self._openai_compat_service = OpenAICompatService(
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
        )

    async def send_message(
        self,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None = None,
    ) -> tuple[str, OpenAIChatCompletionResponse]:
        """处理内部聊天请求，并返回 OpenAI 兼容响应。"""

        request_id = self._generate_identifier()[:8]
        request_start_time = perf_counter()
        execution_request = self._build_execution_request(
            chat_request=chat_request,
            session_id=session_id,
        )
        checkpoint_payload: dict[str, object] | None = None

        try:
            prepare_start_time = perf_counter()
            resolved_session_id = await self._ensure_session(
                session_id=execution_request.session_id,
                user_message=execution_request.latest_user_message,
                user_id=execution_request.user_id,
            )
            execution_request = replace(execution_request, session_id=resolved_session_id)
            await self._persist_user_message(execution_request)
            prepare_duration_ms = (perf_counter() - prepare_start_time) * 1000

            graph_start_time = perf_counter()
            turn_result, checkpoint_payload = await self._conversation_graph.run_turn(
                execution_request
            )
            graph_duration_ms = (perf_counter() - graph_start_time) * 1000

            commit_start_time = perf_counter()
            await self._session_repository.update_timestamp(resolved_session_id)
            await self._db_session.commit()
            commit_duration_ms = (perf_counter() - commit_start_time) * 1000
        except Exception:
            await self._db_session.rollback()
            raise

        checkpoint_start_time = perf_counter()
        await self._save_checkpoint_safely(checkpoint_payload)
        checkpoint_duration_ms = (perf_counter() - checkpoint_start_time) * 1000

        LOGGER.info(
            (
                "聊天请求完成：request_id=%s mode=non_stream session_id=%s "
                "prepare_ms=%.2f graph_ms=%.2f commit_ms=%.2f checkpoint_ms=%.2f "
                "total_ms=%.2f finish_reason=%s"
            ),
            request_id,
            resolved_session_id,
            prepare_duration_ms,
            graph_duration_ms,
            commit_duration_ms,
            checkpoint_duration_ms,
            (perf_counter() - request_start_time) * 1000,
            turn_result.finish_reason,
        )
        return (
            resolved_session_id,
            self._openai_compat_service.build_chat_completion_response(turn_result),
        )

    async def stream_message(
        self,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None = None,
    ) -> tuple[str, AsyncIterator[str]]:
        """处理内部流式聊天请求，并返回 OpenAI 兼容 SSE。"""

        request_id = self._generate_identifier()[:8]
        request_start_time = perf_counter()
        execution_request = self._build_execution_request(
            chat_request=chat_request,
            session_id=session_id,
        )

        try:
            prepare_start_time = perf_counter()
            resolved_session_id = await self._ensure_session(
                session_id=execution_request.session_id,
                user_message=execution_request.latest_user_message,
                user_id=execution_request.user_id,
            )
            execution_request = replace(execution_request, session_id=resolved_session_id)
            await self._persist_user_message(execution_request)
            # 先提交用户消息，避免流式响应已开始时会话仍未落库。
            await self._db_session.commit()
            prepare_duration_ms = (perf_counter() - prepare_start_time) * 1000
        except Exception:
            await self._db_session.rollback()
            raise

        LOGGER.info(
            (
                "聊天流式请求开始：request_id=%s session_id=%s prepare_ms=%.2f"
            ),
            request_id,
            resolved_session_id,
            prepare_duration_ms,
        )
        return (
            resolved_session_id,
            self._consume_graph_events(
                execution_request=execution_request,
                request_id=request_id,
                request_start_time=request_start_time,
            ),
        )

    async def _consume_graph_events(
        self,
        *,
        execution_request: ChatExecutionRequest,
        request_id: str,
        request_start_time: float,
    ) -> AsyncIterator[str]:
        """通用的 LangGraph 事件消息流转化器。"""

        chunk_builder = self._openai_compat_service.create_stream_chunk_builder(
            default_model_name=execution_request.model_name or self._llm_client.default_model_name
        )
        first_payload_duration_ms: float | None = None
        final_result: ChatTurnResult | None = None
        has_emitted_payload = False
        active_node_stack: list[str] = []
        tool_round_chunks: list[AIMessageChunk] = []

        try:
            async for event in self._conversation_graph.stream_events(execution_request):
                event_name = event["event"]
                runnable_name = event.get("name")
                if event_name == "on_chain_start" and runnable_name in _GRAPH_NODE_NAMES:
                    active_node_stack.append(runnable_name)
                elif event_name == "on_chain_end" and runnable_name in _GRAPH_NODE_NAMES:
                    if runnable_name == "tool_node" and tool_round_chunks:
                        for buffered_chunk in tool_round_chunks:
                            for payload in chunk_builder.consume_chunk(buffered_chunk):
                                if first_payload_duration_ms is None:
                                    first_payload_duration_ms = (
                                        perf_counter() - request_start_time
                                    ) * 1000
                                has_emitted_payload = True
                                yield payload
                        tool_round_chunks.clear()
                    if active_node_stack and active_node_stack[-1] == runnable_name:
                        active_node_stack.pop()
                    elif runnable_name in active_node_stack:
                        active_node_stack.remove(runnable_name)

                current_node = active_node_stack[-1] if active_node_stack else None
                # 1. 提取 Token (on_chat_model_stream)
                if event_name == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if not isinstance(chunk, AIMessageChunk):
                        continue

                    if current_node == "tool_node":
                        tool_round_chunks.append(chunk)
                        finish_reason = self._resolve_chunk_finish_reason(chunk)
                        if finish_reason is None:
                            continue

                        if finish_reason == "tool_calls":
                            chunks_to_emit = [
                                filtered_chunk
                                for buffered_chunk in tool_round_chunks
                                if (
                                    filtered_chunk := self._filter_tool_round_chunk_for_tool_calls(
                                        buffered_chunk
                                    )
                                )
                                is not None
                            ]
                        else:
                            chunks_to_emit = list(tool_round_chunks)
                        tool_round_chunks.clear()
                    elif current_node in _USER_VISIBLE_STREAM_NODES:
                        chunks_to_emit = [chunk]
                    else:
                        continue

                    for emit_chunk in chunks_to_emit:
                        for payload in chunk_builder.consume_chunk(emit_chunk):
                            if first_payload_duration_ms is None:
                                first_payload_duration_ms = (
                                    perf_counter() - request_start_time
                                ) * 1000
                            has_emitted_payload = True
                            yield payload

                # 2. 提取最终结果 (on_chain_end for root graph)
                elif event_name == "on_chain_end" and runnable_name == "LangGraph":
                    output = event["data"]["output"]
                    if isinstance(output, dict) and output.get("final_result"):
                        final_result = output["final_result"]

            # 3. 结束流
            for payload in chunk_builder.finalize():
                has_emitted_payload = True
                yield payload

            # 4. 记录日志与指标
            if final_result:
                # 注意：在流式完成后，我们可能需要更新会话时间戳和刷新内存，
                # 但由于 astream_events 已经完成了图运行，这些逻辑应该在 Node 内部或在此处补齐。
                # 目前 AnswerNode 已经处理了持久化，我们只需处理 commit 和 checkpoint。
                await self._session_repository.update_timestamp(str(execution_request.session_id))
                await self._db_session.commit()

                LOGGER.info(
                    (
                        "聊天流式响应完成：request_id=%s session_id=%s route=%s "
                        "first_payload_ms=%s total_elapsed_ms=%.2f"
                    ),
                    request_id,
                    execution_request.session_id,
                    final_result.route,
                    (
                        f"{first_payload_duration_ms:.2f}"
                        if first_payload_duration_ms is not None
                        else "none"
                    ),
                    (perf_counter() - request_start_time) * 1000,
                )

        except AppException as exception:
            await self._db_session.rollback()
            if not has_emitted_payload:
                raise

            yield self._openai_compat_service.build_stream_error_payload(exception)
            yield "data: [DONE]\n\n"
        except Exception as exception:
            await self._db_session.rollback()
            if not has_emitted_payload:
                raise

            LOGGER.exception("流式输出过程中发生未处理异常。", exc_info=exception)
            yield self._openai_compat_service.build_stream_error_payload(
                AppException(
                    "流式输出过程中发生内部异常。",
                    error_code="stream_error",
                )
            )
            yield "data: [DONE]\n\n"

    @staticmethod
    def _resolve_chunk_finish_reason(chunk: AIMessageChunk) -> str | None:
        """从增量块中解析结束原因。"""

        response_metadata = chunk.response_metadata or {}
        finish_reason = response_metadata.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            return finish_reason
        if chunk.tool_calls:
            return "tool_calls"
        return None

    @staticmethod
    def _filter_tool_round_chunk_for_tool_calls(chunk: AIMessageChunk) -> AIMessageChunk | None:
        """tool_calls 轮次仅输出工具调用结构与结束信号。"""

        response_metadata = chunk.response_metadata or {}
        finish_reason = response_metadata.get("finish_reason")
        has_tool_payload = bool(chunk.tool_call_chunks or chunk.tool_calls)
        if not has_tool_payload and finish_reason != "tool_calls":
            return None
        if not has_tool_payload or not chunk.content:
            return chunk
        return chunk.model_copy(update={"content": ""})

    def _build_execution_request(
        self,
        *,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None,
    ) -> ChatExecutionRequest:
        """把 OpenAI 兼容请求转换为内部统一执行请求。"""

        requested_tool_names = self._openai_compat_service.extract_requested_tool_names(
            chat_request
        )
        if requested_tool_names is None and chat_request.tool_choice is not None:
            raise AppException(
                "未传入 tools 时不能指定 tool_choice。",
                error_code="invalid_request",
            )

        return ChatExecutionRequest(
            session_id=session_id,
            need_session_memory=session_id is not None,
            latest_user_message=self._openai_compat_service.extract_latest_user_message(
                chat_request.messages
            ),
            input_messages=self._openai_compat_service.build_input_messages(chat_request.messages),
            model_name=chat_request.model,
            requested_tool_names=requested_tool_names,
            tool_choice=self._tool_registry.normalize_tool_choice(chat_request.tool_choice),
            enable_thinking=chat_request.enable_thinking,
            user_id=chat_request.user,
        )

    async def _ensure_session(
        self,
        *,
        session_id: str | None,
        user_message: str,
        user_id: str | None,
    ) -> str:
        """确保当前请求绑定到可用会话。"""

        if session_id is None:
            session_entity = await self._session_repository.create(
                session_id=self._generate_identifier(),
                title=str(user_message)[:20],
                user_id=user_id,
            )
            return session_entity.session_id

        session_entity = await self._session_repository.get_by_id(session_id)
        if session_entity is None:
            raise ResourceNotFoundException(
                "会话不存在。",
                details={"session_id": session_id},
            )

        return session_id

    async def _persist_user_message(self, execution_request: ChatExecutionRequest) -> None:
        """持久化当前轮次的用户输入。"""

        await self._message_repository.create(
            message_id=self._generate_identifier(),
            session_id=str(execution_request.session_id),
            role="user",
            content=execution_request.latest_user_message,
            message_metadata=execution_request.message_metadata,
        )

    async def _save_checkpoint_safely(
        self,
        checkpoint_payload: dict[str, object] | None,
    ) -> None:
        """在事务提交后保存 checkpoint，并屏蔽非关键基础设施故障。"""

        if checkpoint_payload is None:
            return

        try:
            await self._conversation_graph.save_checkpoint(checkpoint_payload)
        except Exception as exception:  # pragma: no cover - 仅兜底外部基础设施故障
            LOGGER.warning("保存对话 checkpoint 失败，已忽略该异常。", exc_info=exception)

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
