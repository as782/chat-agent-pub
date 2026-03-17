"""对话服务模块。

负责内部聊天接口的会话落库、LangGraph 对话编排，以及 OpenAI 兼容响应构建。
当前阶段负责把非流式和流式链路统一到同一套图状态准备逻辑中，不负责知识库和
MCP 的底层协议实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import ConversationGraph
from app.agent.nodes.tool_node import MAX_TOOL_CALL_ROUNDS
from app.agent.state import AgentState, ChatExecutionRequest
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

        execution_request = self._build_execution_request(
            chat_request=chat_request,
            session_id=session_id,
        )
        checkpoint_payload: dict[str, object] | None = None

        try:
            resolved_session_id = await self._ensure_session(
                session_id=execution_request.session_id,
                user_message=execution_request.latest_user_message,
                user_id=execution_request.user_id,
            )
            execution_request = replace(execution_request, session_id=resolved_session_id)
            await self._persist_user_message(execution_request)
            turn_result, checkpoint_payload = await self._conversation_graph.run_turn(
                execution_request
            )
            await self._session_repository.update_timestamp(resolved_session_id)
            await self._db_session.commit()
        except Exception:
            await self._db_session.rollback()
            raise

        await self._save_checkpoint_safely(checkpoint_payload)
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

        execution_request = self._build_execution_request(
            chat_request=chat_request,
            session_id=session_id,
        )

        try:
            resolved_session_id = await self._ensure_session(
                session_id=execution_request.session_id,
                user_message=execution_request.latest_user_message,
                user_id=execution_request.user_id,
            )
            execution_request = replace(execution_request, session_id=resolved_session_id)
            await self._persist_user_message(execution_request)
            # 先提交用户消息，避免流式响应已开始时会话仍未落库。
            await self._db_session.commit()
            prepared_state = await self._conversation_graph.prepare_stream_state(execution_request)
        except Exception:
            await self._db_session.rollback()
            raise

        return (
            resolved_session_id,
            self._stream_graph_turn(
                execution_request=execution_request,
                prepared_state=prepared_state,
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

    async def _stream_graph_turn(
        self,
        *,
        execution_request: ChatExecutionRequest,
        prepared_state: AgentState,
    ) -> AsyncIterator[str]:
        """在流式路径中执行完整图逻辑，并支持工具/MCP二阶段续答。"""

        route = str(prepared_state["route"])
        prepared_context = prepared_state["prepared_context"]
        chunk_builder = self._openai_compat_service.create_stream_chunk_builder(
            default_model_name=execution_request.model_name or self._llm_client.default_model_name
        )
        answer_node = self._conversation_graph.get_answer_node()
        tool_node = self._conversation_graph.get_tool_node()
        runtime_mcp_tools = tool_node.extract_runtime_mcp_tools(prepared_state)
        available_tools = (
            tool_node.build_available_tools(
                route=route,
                requested_tool_names=execution_request.requested_tool_names,
                runtime_mcp_tools=runtime_mcp_tools,
            )
            if route in {"tool", "mcp"}
            else None
        )
        # 这份对话上下文会在工具执行后持续追加 tool 消息，供下一轮流式补全复用。
        conversation_messages = list(prepared_context.messages)
        normalized_tool_choice = execution_request.tool_choice
        has_emitted_payload = False
        completion_result = None

        try:
            for tool_round in range(MAX_TOOL_CALL_ROUNDS):
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
                    # 中间轮如果模型要求调用工具，不立刻输出 finish 事件，而是继续执行工具后续答。
                    should_emit_finish_reason = not (
                        available_tools is not None and llm_chunk.finish_reason == "tool_calls"
                    )
                    for payload in chunk_builder.consume_chunk(
                        llm_chunk,
                        include_finish_reason=should_emit_finish_reason,
                    ):
                        has_emitted_payload = True
                        yield payload

                completion_result = accumulator.build_result()
                if not completion_result.tool_calls:
                    break

                if available_tools is None:
                    raise AppException(
                        "模型返回了工具调用，但当前请求未开放任何工具。",
                        error_code="invalid_llm_response",
                    )

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
                executed_tool_calls = await tool_node.execute_requested_tools_and_persist(
                    session_id=str(execution_request.session_id),
                    completion_result=completion_result,
                    runtime_mcp_tools=runtime_mcp_tools,
                )
                for executed_tool_call in executed_tool_calls:
                    conversation_messages.append(
                        LlmInputMessage(
                            role="tool",
                            content=executed_tool_call.output,
                            tool_call_id=executed_tool_call.tool_call_id,
                        )
                    )

                normalized_tool_choice = "auto"
                if tool_round + 1 >= MAX_TOOL_CALL_ROUNDS:
                    raise AppException(
                        "工具调用轮次超过当前上限。",
                        error_code="tool_call_limit_exceeded",
                    )

            if completion_result is None:
                raise AppException(
                    "模型未返回最终回答。",
                    error_code="invalid_llm_response",
                )

            await answer_node.persist_stream_result(
                session_id=str(execution_request.session_id),
                completion_result=completion_result,
                used_session_memory=prepared_context.used_session_memory,
            )
            checkpoint_payload = await self._conversation_graph.refresh_memory(
                session_id=str(execution_request.session_id),
                route=route,
            )
            await self._session_repository.update_timestamp(str(execution_request.session_id))
            await self._db_session.commit()
            await self._save_checkpoint_safely(checkpoint_payload)

            for payload in chunk_builder.finalize(completion_result.finish_reason):
                has_emitted_payload = True
                yield payload
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

            LOGGER.exception("内部聊天流式输出过程中发生未处理异常。", exc_info=exception)
            yield self._openai_compat_service.build_stream_error_payload(
                AppException(
                    "流式输出过程中发生内部异常。",
                    error_code="stream_error",
                )
            )
            yield "data: [DONE]\n\n"

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
