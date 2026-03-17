"""对话服务模块。
负责内部聊天接口的会话落库、LangGraph 对话编排与 OpenAI 兼容响应构建。
当前阶段已接入多轮短期记忆，但不负责知识库和 MCP 分支。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import ConversationGraph
from app.agent.state import ChatExecutionRequest, PreparedContext
from app.clients.llm_client import LlmChatCompletionAccumulator, LlmClient
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
        """处理内部聊天请求，并返回 OpenAI 兼容响应与会话标识。"""

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
        """处理内部聊天请求，并返回带多轮记忆的流式 OpenAI 兼容 SSE。"""

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
            # 先提交用户消息与会话，确保流式响应开始后会话已可追踪。
            await self._db_session.commit()
            route, prepared_context = await self._conversation_graph.prepare_stream_context(
                execution_request
            )
        except Exception:
            await self._db_session.rollback()
            raise

        return (
            resolved_session_id,
            self._stream_graph_turn(
                execution_request=execution_request,
                route=route,
                prepared_context=prepared_context,
            ),
        )

    def _build_execution_request(
        self,
        *,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None,
    ) -> ChatExecutionRequest:
        """将 OpenAI 兼容请求转换为内部执行请求。"""

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
        route: str,
        prepared_context: PreparedContext,
    ) -> AsyncIterator[str]:
        """在流式路径中复用上下文构建和记忆刷新逻辑。"""

        chunk_builder = self._openai_compat_service.create_stream_chunk_builder(
            default_model_name=execution_request.model_name or self._llm_client.default_model_name
        )
        accumulator = LlmChatCompletionAccumulator(
            requested_model_name=execution_request.model_name,
            default_model_name=self._llm_client.default_model_name,
        )
        available_tools = (
            self._tool_registry.get_tools(execution_request.requested_tool_names)
            if execution_request.requested_tool_names is not None
            else None
        )
        has_emitted_payload = False

        try:
            async for llm_chunk in self._llm_client.stream_chat_completion(
                messages=prepared_context.messages,
                model_name=execution_request.model_name,
                tools=available_tools,
                tool_choice=execution_request.tool_choice,
                enable_thinking=execution_request.enable_thinking,
            ):
                accumulator.append_chunk(llm_chunk)
                for payload in chunk_builder.consume_chunk(llm_chunk):
                    has_emitted_payload = True
                    yield payload

            completion_result = accumulator.build_result()
            await self._conversation_graph.get_answer_node().persist_stream_result(
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
