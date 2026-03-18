"""对话服务模块。
负责内部聊天接口的会话落库、LangGraph 对话编排与 OpenAI 兼容响应构建。
当前阶段负责把非流式和流式链路统一到同一套图状态准备逻辑中，不负责知识库和 MCP 的底层协议实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import ConversationGraph
from app.agent.nodes.tool_node import MAX_TOOL_CALL_ROUNDS
from app.agent.state import AgentState, ChatExecutionRequest, get_execution_step
from app.clients.llm_client import LlmChatCompletionAccumulator, LlmClient, LlmInputMessage
from app.core.exceptions import AppException, ResourceNotFoundException
from app.core.logger import get_logger
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.openai_compat import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from app.services.openai_compat_service import OpenAICompatService
from app.tools.registry import ToolRegistry

LOGGER = get_logger(__name__)


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
        checkpoint_duration_ms = 0.0

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

            prepare_state_start_time = perf_counter()
            prepared_state = await self._conversation_graph.prepare_stream_state(execution_request)
            prepare_state_duration_ms = (perf_counter() - prepare_state_start_time) * 1000
        except Exception:
            await self._db_session.rollback()
            raise

        LOGGER.info(
            (
                "聊天请求开始：request_id=%s mode=stream session_id=%s route=%s "
                "prepare_ms=%.2f prepare_state_ms=%.2f total_elapsed_ms=%.2f"
            ),
            request_id,
            resolved_session_id,
            str(prepared_state["route"]),
            prepare_duration_ms,
            prepare_state_duration_ms,
            (perf_counter() - request_start_time) * 1000,
        )
        return (
            resolved_session_id,
            self._stream_planned_graph_turn(
                execution_request=execution_request,
                prepared_state=prepared_state,
                request_id=request_id,
                request_start_time=request_start_time,
            ),
        )

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

    async def _stream_planned_graph_turn(
        self,
        *,
        execution_request: ChatExecutionRequest,
        prepared_state: AgentState,
        request_id: str,
        request_start_time: float,
    ) -> AsyncIterator[str]:
        """执行支持多 step 调度的流式对话主逻辑。"""

        request_route = str(prepared_state["route"])
        current_route = request_route
        working_state = prepared_state
        chunk_builder = self._openai_compat_service.create_stream_chunk_builder(
            default_model_name=execution_request.model_name or self._llm_client.default_model_name
        )
        answer_node = self._conversation_graph.get_answer_node()
        tool_node = self._conversation_graph.get_tool_node()
        has_emitted_payload = False
        final_completion_result = None
        final_used_session_memory = prepared_state["prepared_context"].used_session_memory
        first_payload_duration_ms: float | None = None
        tool_execution_duration_ms = 0.0
        persist_duration_ms = 0.0
        memory_refresh_duration_ms = 0.0
        commit_duration_ms = 0.0
        checkpoint_duration_ms = 0.0

        try:
            while True:
                current_route = str(working_state["route"])
                prepared_context = working_state["prepared_context"]
                final_used_session_memory = prepared_context.used_session_memory

                if self._is_tool_route(current_route):
                    runtime_mcp_tools = tool_node.extract_runtime_mcp_tools(working_state)
                    available_tools = tool_node.build_available_tools(
                        route=current_route,
                        requested_tool_names=execution_request.requested_tool_names,
                        runtime_mcp_tools=runtime_mcp_tools,
                    )
                    conversation_messages = list(prepared_context.messages)
                    normalized_tool_choice = execution_request.tool_choice
                    suppress_step_content = self._should_defer_stream_step_content(working_state)
                    executed_tool_calls = []
                    completion_result = None

                    for tool_round in range(MAX_TOOL_CALL_ROUNDS):
                        round_index = tool_round + 1
                        llm_round_start_time = perf_counter()
                        accumulator = LlmChatCompletionAccumulator(
                            requested_model_name=execution_request.model_name,
                            default_model_name=self._llm_client.default_model_name,
                        )
                        async for llm_chunk in self._llm_client.stream_chat_completion(
                            messages=conversation_messages,
                            model_name=execution_request.model_name,
                            tools=available_tools,
                            tool_choice=normalized_tool_choice,
                            enable_thinking=execution_request.enable_thinking,
                        ):
                            accumulator.append_chunk(llm_chunk)
                            if suppress_step_content and not llm_chunk.tool_call_chunks:
                                continue

                            should_emit_finish_reason = llm_chunk.finish_reason != "tool_calls"
                            for payload in chunk_builder.consume_chunk(
                                llm_chunk,
                                include_finish_reason=should_emit_finish_reason,
                            ):
                                first_payload_duration_ms = self._mark_first_stream_payload(
                                    request_id=request_id,
                                    route=request_route,
                                    request_start_time=request_start_time,
                                    current_first_payload_duration_ms=first_payload_duration_ms,
                                )
                                has_emitted_payload = True
                                yield payload

                        completion_result = accumulator.build_result()
                        LOGGER.info(
                            (
                                "聊天流式轮次完成：request_id=%s route=%s round=%s "
                                "llm_round_ms=%.2f finish_reason=%s tool_call_count=%s"
                            ),
                            request_id,
                            current_route,
                            round_index,
                            (perf_counter() - llm_round_start_time) * 1000,
                            completion_result.finish_reason,
                            len(completion_result.tool_calls),
                        )
                        if not completion_result.tool_calls:
                            break

                        await answer_node.persist_assistant_tool_calls(
                            session_id=str(execution_request.session_id),
                            completion_result=completion_result,
                        )
                        conversation_messages.append(
                            LlmInputMessage(
                                role="assistant",
                                content=completion_result.content,
                                tool_calls=completion_result.tool_calls,
                            )
                        )

                        tool_execute_start_time = perf_counter()
                        current_tool_results = await tool_node.execute_requested_tools_and_persist(
                            session_id=str(execution_request.session_id),
                            completion_result=completion_result,
                            runtime_mcp_tools=runtime_mcp_tools,
                        )
                        current_tool_execution_duration_ms = (
                            perf_counter() - tool_execute_start_time
                        ) * 1000
                        tool_execution_duration_ms += current_tool_execution_duration_ms
                        LOGGER.info(
                            (
                                "聊天流式工具执行完成：request_id=%s route=%s round=%s "
                                "tool_exec_ms=%.2f executed_tool_count=%s"
                            ),
                            request_id,
                            current_route,
                            round_index,
                            current_tool_execution_duration_ms,
                            len(current_tool_results),
                        )
                        executed_tool_calls.extend(current_tool_results)
                        for executed_tool_call in current_tool_results:
                            conversation_messages.append(
                                LlmInputMessage(
                                    role="tool",
                                    content=executed_tool_call.output,
                                    tool_call_id=executed_tool_call.tool_call_id,
                                )
                            )

                        normalized_tool_choice = "auto"
                        if round_index >= MAX_TOOL_CALL_ROUNDS:
                            raise AppException(
                                "Tool call rounds exceeded the current limit.",
                                error_code="tool_call_limit_exceeded",
                            )

                    if completion_result is None:
                        raise AppException(
                            "LLM did not return a final response.",
                            error_code="invalid_llm_response",
                        )

                    working_state = {
                        **working_state,
                        "tool_completion_result": completion_result,
                        "executed_tool_calls": executed_tool_calls,
                        **tool_node.build_step_result_update(
                            state=working_state,
                            completion_result=completion_result,
                            executed_tool_calls=executed_tool_calls,
                        ),
                    }
                    advanced_state = await self._conversation_graph.advance_stream_state(
                        working_state
                    )
                    if str(
                        advanced_state["route"]
                    ) == "answer" and not answer_node.should_generate_summary(advanced_state):
                        working_state = advanced_state
                        final_used_session_memory = advanced_state[
                            "prepared_context"
                        ].used_session_memory
                        final_completion_result = completion_result
                        break

                    working_state = advanced_state
                    continue

                llm_round_start_time = perf_counter()
                accumulator = LlmChatCompletionAccumulator(
                    requested_model_name=execution_request.model_name,
                    default_model_name=self._llm_client.default_model_name,
                )
                async for llm_chunk in self._llm_client.stream_chat_completion(
                    messages=prepared_context.messages,
                    model_name=execution_request.model_name,
                    enable_thinking=execution_request.enable_thinking,
                ):
                    accumulator.append_chunk(llm_chunk)
                    for payload in chunk_builder.consume_chunk(llm_chunk):
                        first_payload_duration_ms = self._mark_first_stream_payload(
                            request_id=request_id,
                            route=request_route,
                            request_start_time=request_start_time,
                            current_first_payload_duration_ms=first_payload_duration_ms,
                        )
                        has_emitted_payload = True
                        yield payload

                final_completion_result = accumulator.build_result()
                LOGGER.info(
                    (
                        "聊天流式轮次完成：request_id=%s route=%s round=%s "
                        "llm_round_ms=%.2f finish_reason=%s tool_call_count=%s"
                    ),
                    request_id,
                    current_route,
                    1,
                    (perf_counter() - llm_round_start_time) * 1000,
                    final_completion_result.finish_reason,
                    len(final_completion_result.tool_calls),
                )
                break

            if final_completion_result is None:
                raise AppException(
                    "LLM did not return a final response.",
                    error_code="invalid_llm_response",
                )

            persist_start_time = perf_counter()
            await answer_node.persist_stream_result(
                session_id=str(execution_request.session_id),
                completion_result=final_completion_result,
                used_session_memory=final_used_session_memory,
            )
            persist_duration_ms = (perf_counter() - persist_start_time) * 1000

            memory_refresh_start_time = perf_counter()
            checkpoint_payload = await self._conversation_graph.refresh_memory(
                session_id=str(execution_request.session_id),
                route=request_route,
            )
            memory_refresh_duration_ms = (perf_counter() - memory_refresh_start_time) * 1000

            commit_start_time = perf_counter()
            await self._session_repository.update_timestamp(str(execution_request.session_id))
            await self._db_session.commit()
            commit_duration_ms = (perf_counter() - commit_start_time) * 1000

            checkpoint_start_time = perf_counter()
            await self._save_checkpoint_safely(checkpoint_payload)
            checkpoint_duration_ms = (perf_counter() - checkpoint_start_time) * 1000

            for payload in chunk_builder.finalize(final_completion_result.finish_reason):
                first_payload_duration_ms = self._mark_first_stream_payload(
                    request_id=request_id,
                    route=request_route,
                    request_start_time=request_start_time,
                    current_first_payload_duration_ms=first_payload_duration_ms,
                )
                has_emitted_payload = True
                yield payload

            LOGGER.info(
                (
                    "聊天请求完成：request_id=%s mode=stream session_id=%s route=%s "
                    "first_payload_ms=%s tool_exec_ms=%.2f persist_ms=%.2f "
                    "memory_refresh_ms=%.2f commit_ms=%.2f checkpoint_ms=%.2f "
                    "total_ms=%.2f finish_reason=%s"
                ),
                request_id,
                execution_request.session_id,
                request_route,
                f"{first_payload_duration_ms:.2f}"
                if first_payload_duration_ms is not None
                else "none",
                tool_execution_duration_ms,
                persist_duration_ms,
                memory_refresh_duration_ms,
                commit_duration_ms,
                checkpoint_duration_ms,
                (perf_counter() - request_start_time) * 1000,
                final_completion_result.finish_reason,
            )
        except AppException as exception:
            await self._db_session.rollback()
            if not has_emitted_payload:
                raise

            LOGGER.warning(
                (
                    "聊天请求失败：request_id=%s mode=stream session_id=%s route=%s "
                    "total_ms=%.2f error_code=%s"
                ),
                request_id,
                execution_request.session_id,
                request_route,
                (perf_counter() - request_start_time) * 1000,
                exception.error_code,
            )
            yield self._openai_compat_service.build_stream_error_payload(exception)
            yield "data: [DONE]\n\n"
        except Exception as exception:
            await self._db_session.rollback()
            if not has_emitted_payload:
                raise

            LOGGER.exception(
                "内部聊天流式输出过程中发生未处理异常：request_id=%s route=%s",
                request_id,
                current_route,
                exc_info=exception,
            )
            yield self._openai_compat_service.build_stream_error_payload(
                AppException(
                    "Internal stream error.",
                    error_code="stream_error",
                )
            )
            yield "data: [DONE]\n\n"

    @staticmethod
    def _is_tool_route(route: str) -> bool:
        """判断当前路由是否需要进入工具/MCP 取数合并回答的技术分支。"""

        return route in {"tool", "route", "mcp", "traffic", "report"}

    @staticmethod
    def _should_defer_stream_step_content(state: AgentState) -> bool:
        """Suppress intermediate natural-language output for multi-step executor plans."""

        execution_plan = state.get("execution_plan")
        current_step = get_execution_step(
            state,
            step_id=(
                str(state["current_step_id"]) if state.get("current_step_id") is not None else None
            ),
        )
        if execution_plan is None or current_step is None or current_step.executor == "answer":
            return False

        non_answer_step_count = sum(1 for step in execution_plan.steps if step.executor != "answer")
        return non_answer_step_count > 1

    def _mark_first_stream_payload(
        self,
        *,
        request_id: str,
        route: str,
        request_start_time: float,
        current_first_payload_duration_ms: float | None,
    ) -> float:
        """记录第一次 SSE 输出的时间戳，用于后续日志记录。"""

        if current_first_payload_duration_ms is not None:
            return current_first_payload_duration_ms

        first_payload_duration_ms = (perf_counter() - request_start_time) * 1000
        LOGGER.info(
            (
                "聊天请求第一次输出：request_id=%s "
                "route=%s first_payload_ms=%.2f"
            ),
            request_id,
            route,
            first_payload_duration_ms,
        )
        return first_payload_duration_ms

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
                title=user_message[:20],
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

        try:
            await self._conversation_graph.save_checkpoint(checkpoint_payload)
        except Exception as exception:  # pragma: no cover - 仅兜底外部基础设施故障
            LOGGER.warning("保存对话 checkpoint 失败，已忽略该异常。", exc_info=exception)

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
