"""对话服务模块。
负责内部聊天接口的会话落库、LangGraph 对话编排与 OpenAI 兼容响应构建。
当前阶段负责把非流式和流式链路统一到同一套图状态准备逻辑中，不负责知识库和 MCP 的底层协议实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessageChunk
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import ConversationGraph
from app.agent.state import (
    ChatExecutionRequest,
    ChatTurnResult,
)
from app.clients.llm_client import LlmClient
from app.core.config import get_settings
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
    "service_node",
    "report_node",
    "answer_node",
    "memory_node",
}
_USER_VISIBLE_STREAM_NODES = {"answer_node"}
_ROOT_GRAPH_RUNNABLE_NAME = "LangGraph"
_STREAM_NODE_ROLE_MAPPING = {
    "answer_node": "assistant",
    "tool_node": "assistant",
}


@dataclass(slots=True)
class _PreparedChatExecution:
    """封装聊天请求准备阶段的结果。"""

    request_id: str
    request_start_time: float
    execution_request: ChatExecutionRequest
    resolved_session_id: str
    prepare_duration_ms: float


@dataclass(slots=True)
class _GraphStreamState:
    """封装图流式处理过程中的中间状态。"""

    first_payload_duration_ms: float | None = None
    final_result: ChatTurnResult | None = None
    checkpoint_payload: dict[str, object] | None = None
    has_emitted_payload: bool = False
    emitted_content: str = ""
    active_node_stack: list[str] = field(default_factory=list)
    visited_nodes: list[str] = field(default_factory=list)
    tool_round_chunks: list[AIMessageChunk] = field(default_factory=list) 


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

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex

    @staticmethod
    def _enter_graph_node(stream_state: _GraphStreamState, runnable_name: str) -> None:
        """记录进入图节点的事件。"""

        stream_state.active_node_stack.append(runnable_name)
        if runnable_name not in stream_state.visited_nodes:
            stream_state.visited_nodes.append(runnable_name)

    @staticmethod
    def _exit_graph_node(stream_state: _GraphStreamState, runnable_name: str) -> None:
        """记录离开图节点的事件。"""

        if stream_state.active_node_stack and stream_state.active_node_stack[-1] == runnable_name:
            stream_state.active_node_stack.pop()
            return
        if runnable_name in stream_state.active_node_stack:
            stream_state.active_node_stack.remove(runnable_name)

    @staticmethod
    def _get_current_graph_node(stream_state: _GraphStreamState) -> str | None:
        """获取当前活跃的图节点。"""

        return stream_state.active_node_stack[-1] if stream_state.active_node_stack else None

    def _flush_tool_round_chunks(
        self,
        stream_state: _GraphStreamState,
        *,
        tool_calls_only: bool,
    ) -> list[AIMessageChunk]:
        """刷新当前 tool_node 轮次中缓冲的 chunk。"""

        if not stream_state.tool_round_chunks:
            return []

        if tool_calls_only:
            chunks_to_emit = [
                filtered_chunk
                for buffered_chunk in stream_state.tool_round_chunks
                if (
                    filtered_chunk := self._filter_tool_round_chunk_for_tool_calls(buffered_chunk)
                )
                is not None
            ]
        else:
            chunks_to_emit = list(stream_state.tool_round_chunks)
        stream_state.tool_round_chunks.clear()
        return chunks_to_emit

    def _collect_chunks_from_stream_event(
        self,
        *,
        stream_state: _GraphStreamState,
        current_node: str | None,
        event: dict[str, Any],
    ) -> list[AIMessageChunk]:
        """根据单个 LangGraph 事件提取需要对外输出的 chunk。"""

        if event["event"] != "on_chat_model_stream":
            return []

        chunk = event["data"]["chunk"]
        if not isinstance(chunk, AIMessageChunk):
            return []

        if current_node == "tool_node":
            stream_state.tool_round_chunks.append(chunk)
            finish_reason = self._resolve_chunk_finish_reason(chunk)
            if finish_reason is None:
                return []
            return self._flush_tool_round_chunks(
                stream_state,
                tool_calls_only=finish_reason == "tool_calls",
            )

        if current_node in _USER_VISIBLE_STREAM_NODES:
            return [chunk]
        return []

    @staticmethod
    def _resolve_stream_chunk_role(current_node: str | None) -> str:
        """根据当前图节点解析对外流式 chunk 的消息角色。"""

        return _STREAM_NODE_ROLE_MAPPING.get(current_node or "", "assistant")

    @staticmethod
    def _extract_chunk_text(chunk: AIMessageChunk) -> str:
        """把流式 chunk 的内容归一化为纯文本。"""

        if isinstance(chunk.content, str):
            return chunk.content
        if isinstance(chunk.content, list):
            text_parts: list[str] = []
            for part in chunk.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return str(chunk.content or "")

    async def _emit_stream_payloads(
        self,
        *,
        stream_state: _GraphStreamState,
        request_start_time: float,
        chunk_builder: Any,
        chunks: list[AIMessageChunk],
        role: str,
        track_emitted_content: bool = False,
        include_finish_reason: bool = True,
    ) -> AsyncIterator[str]:
        """把 chunk 列表统一转换为 OpenAI 兼容 SSE。"""

        for emit_chunk in chunks:
            if track_emitted_content:
                stream_state.emitted_content += self._extract_chunk_text(emit_chunk)
            for payload in chunk_builder.consume_chunk(
                emit_chunk,
                role=role,
                include_finish_reason=include_finish_reason,
            ):
                if stream_state.first_payload_duration_ms is None:
                    stream_state.first_payload_duration_ms = (
                        perf_counter() - request_start_time
                    ) * 1000
                stream_state.has_emitted_payload = True
                yield payload

    @staticmethod
    def _capture_graph_completion(
        stream_state: _GraphStreamState,
        *,
        output: object,
    ) -> None:
        """从根图输出中提取最终结果与 checkpoint。"""

        if not isinstance(output, Mapping):
            return

        final_result = output.get("final_result")
        normalized_final_result = ChatService._coerce_chat_turn_result(final_result)
        if normalized_final_result is not None:
            stream_state.final_result = normalized_final_result

        checkpoint_payload = output.get("checkpoint_payload")
        if isinstance(checkpoint_payload, dict):
            stream_state.checkpoint_payload = checkpoint_payload

    @staticmethod
    def _coerce_chat_turn_result(value: object) -> ChatTurnResult | None:
        """Normalize graph outputs that may serialize `ChatTurnResult` into a plain dict."""

        if isinstance(value, ChatTurnResult):
            return replace(
                value,
                content=ChatService._sanitize_network_report_content(value.content),
            )
        if not isinstance(value, Mapping):
            return None

        content = value.get("content")
        model_name = value.get("model_name")
        finish_reason = value.get("finish_reason")
        if not isinstance(content, str) or not content.strip():
            return None

        try:
            prompt_tokens = int(value.get("prompt_tokens") or 0)
            completion_tokens = int(value.get("completion_tokens") or 0)
            total_tokens = int(value.get("total_tokens") or (prompt_tokens + completion_tokens))
        except (TypeError, ValueError):
            return None

        return ChatTurnResult(
            session_id=str(value.get("session_id") or ""),
            content=ChatService._sanitize_network_report_content(content),
            model_name=str(model_name or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(finish_reason or "stop"),
            route=str(value.get("route") or "answer"),
            reasoning_content=(
                str(value["reasoning_content"])
                if isinstance(value.get("reasoning_content"), str)
                and str(value.get("reasoning_content")).strip()
                else None
            ),
            tool_calls=[],
            used_session_memory=bool(value.get("used_session_memory")),
        )

    async def _prepare_chat_execution(
        self,
        *,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None,
        commit_after_prepare: bool,
    ) -> _PreparedChatExecution:
        """统一处理请求准备阶段的会话解析与用户消息落库。"""

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
            if commit_after_prepare:
                await self._db_session.commit()
            prepare_duration_ms = (perf_counter() - prepare_start_time) * 1000
        except Exception:
            await self._db_session.rollback()
            raise

        return _PreparedChatExecution(
            request_id=request_id,
            request_start_time=request_start_time,
            execution_request=execution_request,
            resolved_session_id=resolved_session_id,
            prepare_duration_ms=prepare_duration_ms,
        )

    async def _consume_graph_events(
        self,
        *,
        execution_request: ChatExecutionRequest,
        request_id: str,
        request_start_time: float,
        prepare_duration_ms: float,
    ) -> AsyncIterator[str]:
        """通用的 LangGraph 事件消息流转化器。"""

        chunk_builder = self._openai_compat_service.create_stream_chunk_builder(
            default_model_name=execution_request.model_name or self._llm_client.default_model_name
        )
        stream_state = _GraphStreamState()
        graph_start_time = perf_counter()

        try:
            async for event in self._conversation_graph.stream_events(execution_request):
                event_name = event["event"]
                runnable_name = event.get("name")

                if event_name == "on_chain_start" and runnable_name in _GRAPH_NODE_NAMES:
                    self._enter_graph_node(stream_state, runnable_name)
                elif event_name == "on_chain_end" and runnable_name in _GRAPH_NODE_NAMES:
                    chunks_to_emit = (
                        self._flush_tool_round_chunks(
                            stream_state,
                            tool_calls_only=False,
                        )
                        if runnable_name == "tool_node"
                        else []
                    )
                    self._exit_graph_node(stream_state, runnable_name)
                    async for payload in self._emit_stream_payloads(
                        stream_state=stream_state,
                        request_start_time=request_start_time,
                        chunk_builder=chunk_builder,
                        chunks=chunks_to_emit,
                        role=self._resolve_stream_chunk_role(runnable_name),
                        track_emitted_content=False,
                        include_finish_reason=True,
                    ):
                        yield payload

                current_node = self._get_current_graph_node(stream_state)
                chunks_to_emit = self._collect_chunks_from_stream_event(
                    stream_state=stream_state,
                    current_node=current_node,
                    event=event,
                )
                async for payload in self._emit_stream_payloads(
                    stream_state=stream_state,
                    request_start_time=request_start_time,
                    chunk_builder=chunk_builder,
                    chunks=chunks_to_emit,
                    role=self._resolve_stream_chunk_role(current_node),
                    track_emitted_content=current_node in _USER_VISIBLE_STREAM_NODES,
                    include_finish_reason=current_node not in _USER_VISIBLE_STREAM_NODES,
                ):
                    yield payload

                if event_name == "on_chain_end" and runnable_name == _ROOT_GRAPH_RUNNABLE_NAME:
                    self._capture_graph_completion(
                        stream_state,
                        output=event["data"]["output"],
                    )

            async for payload in self._emit_final_result_payload_if_needed(
                stream_state=stream_state,
                request_start_time=request_start_time,
                chunk_builder=chunk_builder,
            ):
                yield payload

            for payload in chunk_builder.finalize():
                stream_state.has_emitted_payload = True
                yield payload

            if stream_state.final_result:
                graph_duration_ms = (perf_counter() - graph_start_time) * 1000
                commit_start_time = perf_counter()
                await self._session_repository.update_timestamp(str(execution_request.session_id))
                await self._db_session.commit()
                commit_duration_ms = (perf_counter() - commit_start_time) * 1000

                checkpoint_start_time = perf_counter()
                await self._save_checkpoint_safely(stream_state.checkpoint_payload)
                checkpoint_duration_ms = (perf_counter() - checkpoint_start_time) * 1000

                LOGGER.info(
                    (
                        "聊天请求完成：request_id=%s mode=stream session_id=%s "
                        "prepare_ms=%.2f graph_ms=%.2f commit_ms=%.2f checkpoint_ms=%.2f "
                        "total_ms=%.2f finish_reason=%s visited_nodes=%s first_payload_ms=%s"
                    ),
                    request_id,
                    execution_request.session_id,
                    prepare_duration_ms,
                    graph_duration_ms,
                    commit_duration_ms,
                    checkpoint_duration_ms,
                    (perf_counter() - request_start_time) * 1000,
                    stream_state.final_result.finish_reason,
                    "->".join(stream_state.visited_nodes) if stream_state.visited_nodes else "none",
                    (
                        f"{stream_state.first_payload_duration_ms:.2f}"
                        if stream_state.first_payload_duration_ms is not None
                        else "none"
                    ),
                )

        except AppException as exception:
            await self._db_session.rollback()
            if not stream_state.has_emitted_payload:
                raise

            yield self._openai_compat_service.build_stream_error_payload(exception)
            yield "data: [DONE]\n\n"
        except Exception as exception:
            await self._db_session.rollback()
            if not stream_state.has_emitted_payload:
                raise

            LOGGER.exception("流式输出过程中发生未处理异常。", exc_info=exception)
            yield self._openai_compat_service.build_stream_error_payload(
                AppException(
                    "流式输出过程中发生内部异常。",
                    error_code="stream_error",
            )
        )

    @staticmethod
    def _sanitize_network_report_content(content: str) -> str:
        normalized_content = content.strip()
        for prefix in (
            "AI播报总结：",
            "AI播报总结:",
            "播报总结：",
            "播报总结:",
            "总结：",
            "总结:",
        ):
            if normalized_content.startswith(prefix):
                normalized_content = normalized_content[len(prefix) :].lstrip()
        return normalized_content
    async def _emit_final_result_payload_if_needed(
        self,
        *,
        stream_state: _GraphStreamState,
        request_start_time: float,
        chunk_builder: object,
    ) -> AsyncIterator[str]:
        """在流结束时，把 final_result 中尚未发出的尾部内容补发出来。"""

        final_result = stream_state.final_result
        if final_result is None:
            return

        final_content = str(final_result.content or "")
        if not final_content:
            return

        emitted_content = stream_state.emitted_content
        if emitted_content and final_content.startswith(emitted_content):
            payload_content = final_content[len(emitted_content) :]
        elif emitted_content:
            payload_content = self._extract_final_result_suffix_for_stream(
                final_content=final_content,
                emitted_content=emitted_content,
            )
            if payload_content is None:
                return
        elif not stream_state.has_emitted_payload:
            payload_content = final_content
        else:
            return

        if not payload_content:
            return

        if stream_state.first_payload_duration_ms is None:
            stream_state.first_payload_duration_ms = (
                perf_counter() - request_start_time
            ) * 1000
        stream_state.has_emitted_payload = True
        yield chunk_builder._build_content_payload(payload_content)

    @staticmethod
    def _extract_final_result_suffix_for_stream(
        *,
        final_content: str,
        emitted_content: str,
    ) -> str | None:
        """Handle report summaries whose final_result adds a prefix after streaming text."""

        for prefix in (
            "AI播报总结：",
            "AI播报总结:",
            "播报总结：",
            "播报总结:",
        ):
            if not final_content.startswith(prefix):
                continue
            stripped_content = final_content[len(prefix) :].lstrip("\n")
            if stripped_content.startswith(emitted_content):
                return stripped_content[len(emitted_content) :]
        return None

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
            model_name=(
                self._llm_client.default_model_name
                if get_settings().use_monitor_network_development_upstreams
                else chat_request.model
            ),
            requested_tool_names=requested_tool_names,
            tool_choice=self._tool_registry.normalize_tool_choice(chat_request.tool_choice),
            scheduled_route=chat_request.scheduled_route,
            enable_thinking=chat_request.resolved_enable_thinking,
            brief_answer=chat_request.brief_answer,
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

    async def send_message(
        self,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None = None,
    ) -> tuple[str, OpenAIChatCompletionResponse]:
        """处理内部聊天请求，并返回 OpenAI 兼容响应。"""

        prepared_execution = await self._prepare_chat_execution(
            chat_request=chat_request,
            session_id=session_id,
            commit_after_prepare=False,
        )
        checkpoint_payload: dict[str, object] | None = None

        try:
            graph_start_time = perf_counter()
            turn_result, checkpoint_payload = await self._conversation_graph.run_turn(
                prepared_execution.execution_request
            )
            graph_duration_ms = (perf_counter() - graph_start_time) * 1000

            commit_start_time = perf_counter()
            await self._session_repository.update_timestamp(prepared_execution.resolved_session_id)
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
            prepared_execution.request_id,
            prepared_execution.resolved_session_id,
            prepared_execution.prepare_duration_ms,
            graph_duration_ms,
            commit_duration_ms,
            checkpoint_duration_ms,
            (perf_counter() - prepared_execution.request_start_time) * 1000,
            turn_result.finish_reason,
        )
        return (
            prepared_execution.resolved_session_id,
            self._openai_compat_service.build_chat_completion_response(turn_result),
        )

    async def stream_message(
        self,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None = None,
    ) -> tuple[str, AsyncIterator[str]]:
        """处理内部流式聊天请求，并返回 OpenAI 兼容 SSE。"""

        prepared_execution = await self._prepare_chat_execution(
            chat_request=chat_request,
            session_id=session_id,
            commit_after_prepare=True,
        )

        LOGGER.info(
            "聊天流式请求开始：request_id=%s session_id=%s prepare_ms=%.2f scheduled_route=%s",
            prepared_execution.request_id,
            prepared_execution.resolved_session_id,
            prepared_execution.prepare_duration_ms,
            prepared_execution.execution_request.scheduled_route,
        )
        return (
            prepared_execution.resolved_session_id,
            self._consume_graph_events(
                execution_request=prepared_execution.execution_request,
                request_id=prepared_execution.request_id,
                request_start_time=prepared_execution.request_start_time,
                prepare_duration_ms=prepared_execution.prepare_duration_ms,
            ),
        )
