"""对话服务模块。
负责内部聊天接口的会话落库、工具执行与 OpenAI 兼容响应编排。
当前阶段不负责多轮长期记忆、LangGraph 状态图和知识库路由决策。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.llm_client import LlmChatCompletionResult, LlmClient, LlmInputMessage
from app.core.exceptions import AppException, ResourceNotFoundException
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.openai_compat import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from app.services.openai_compat_service import OpenAICompatService
from app.tools.registry import ExecutedToolCall, ToolRegistry

MAX_TOOL_CALL_ROUNDS = 3


@dataclass(slots=True)
class ChatExecutionRequest:
    """内部聊天执行请求。"""

    session_id: str | None
    latest_user_message: str
    input_messages: list[LlmInputMessage]
    model_name: str | None
    requested_tool_names: list[str] | None
    tool_choice: str | dict[str, object] | None
    user_id: str | None = None
    message_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ChatTurnResult:
    """单轮对话执行结果。"""

    session_id: str
    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)


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
        turn_result = await self._execute_chat_turn(execution_request)
        return (
            turn_result.session_id,
            self._openai_compat_service.build_chat_completion_response(turn_result),
        )

    async def stream_message(
        self,
        chat_request: OpenAIChatCompletionRequest,
        session_id: str | None = None,
    ) -> tuple[str, AsyncIterator[str]]:
        """处理内部聊天请求，并返回 OpenAI 兼容流式迭代器与会话标识。"""

        execution_request = self._build_execution_request(
            chat_request=chat_request,
            session_id=session_id,
        )
        turn_result = await self._execute_chat_turn(execution_request)
        return (
            turn_result.session_id,
            self._openai_compat_service.build_stream_chat_completion(turn_result),
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
            latest_user_message=self._openai_compat_service.extract_latest_user_message(
                chat_request.messages
            ),
            input_messages=self._openai_compat_service.build_input_messages(chat_request.messages),
            model_name=chat_request.model,
            requested_tool_names=requested_tool_names,
            tool_choice=self._tool_registry.normalize_tool_choice(chat_request.tool_choice),
            user_id=chat_request.user,
        )

    async def _execute_chat_turn(
        self,
        execution_request: ChatExecutionRequest,
    ) -> ChatTurnResult:
        """执行单轮对话与工具调用流程。"""

        try:
            session_id = await self._ensure_session(
                session_id=execution_request.session_id,
                user_message=execution_request.latest_user_message,
                user_id=execution_request.user_id,
            )
            await self._message_repository.create(
                message_id=self._generate_identifier(),
                session_id=session_id,
                role="user",
                content=execution_request.latest_user_message,
                message_metadata=execution_request.message_metadata,
            )

            conversation_messages = list(execution_request.input_messages)
            available_tools = (
                self._tool_registry.get_tools(execution_request.requested_tool_names)
                if execution_request.requested_tool_names is not None
                else None
            )
            tool_choice = execution_request.tool_choice

            executed_tool_calls: list[ExecutedToolCall] = []
            final_result: LlmChatCompletionResult | None = None

            for tool_round in range(MAX_TOOL_CALL_ROUNDS):
                completion_result = await self._llm_client.create_chat_completion(
                    messages=conversation_messages,
                    model_name=execution_request.model_name,
                    tools=available_tools,
                    tool_choice=tool_choice,
                )

                if completion_result.tool_calls:
                    await self._persist_assistant_tool_calls(
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
                            message_id=self._generate_identifier(),
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

                    tool_choice = "auto"
                    if tool_round + 1 >= MAX_TOOL_CALL_ROUNDS:
                        raise AppException(
                            "工具调用轮次超过当前上限。",
                            error_code="tool_call_limit_exceeded",
                        )
                    continue

                final_result = completion_result
                break

            if final_result is None:
                raise AppException(
                    "模型未返回最终回答。",
                    error_code="invalid_llm_response",
                )

            await self._message_repository.create(
                message_id=self._generate_identifier(),
                session_id=session_id,
                role="assistant",
                content=final_result.content,
                message_metadata={
                    "finish_reason": final_result.finish_reason,
                    "model_name": final_result.model_name,
                },
            )
            await self._session_repository.update_timestamp(session_id)
            await self._db_session.commit()
        except Exception:
            await self._db_session.rollback()
            raise

        return ChatTurnResult(
            session_id=session_id,
            content=final_result.content,
            model_name=final_result.model_name,
            prompt_tokens=final_result.prompt_tokens,
            completion_tokens=final_result.completion_tokens,
            total_tokens=final_result.total_tokens,
            finish_reason=final_result.finish_reason,
            tool_calls=executed_tool_calls,
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

    async def _persist_assistant_tool_calls(
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
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
