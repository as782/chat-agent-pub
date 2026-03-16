"""对话服务模块。
负责基础单轮对话、内置工具执行、消息落库与流式输出编排。
当前阶段不负责多轮长期记忆、LangGraph 状态图和知识库路由决策。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from json import dumps
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.llm_client import LlmChatCompletionResult, LlmClient, LlmInputMessage
from app.core.exceptions import AppException, ResourceNotFoundException
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.chat import ChatRequest, ChatResponse, ChatToolCallResponse
from app.tools.registry import ExecutedToolCall, ToolRegistry

MAX_TOOL_CALL_ROUNDS = 3
DEFAULT_SYSTEM_PROMPT = "你是最小可用 Agent 后端中的基础问答模块，需要简洁、准确地回答用户。"


@dataclass(slots=True)
class ChatTurnResult:
    """单轮对话执行结果。"""

    session_id: str
    answer: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    used_tools: list[str] = field(default_factory=list)
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)


class ChatService:
    """基础单轮对话服务。"""

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

    async def send_message(self, chat_request: ChatRequest) -> ChatResponse:
        """处理单轮对话请求并持久化本轮消息。"""

        turn_result = await self._execute_chat_turn(chat_request)
        return self._build_chat_response(turn_result)

    async def stream_message(self, chat_request: ChatRequest) -> AsyncIterator[str]:
        """以 SSE 形式输出内部聊天接口的流式结果。"""

        turn_result = await self._execute_chat_turn(chat_request)

        yield self._format_sse_payload(
            {
                "type": "message_start",
                "session_id": turn_result.session_id,
                "model": turn_result.model_name,
            }
        )

        for tool_call in turn_result.tool_calls:
            yield self._format_sse_payload(
                {
                    "type": "tool_call",
                    "tool_call_id": tool_call.tool_call_id,
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output": tool_call.output,
                }
            )

        answer_chunks = self._split_text_for_stream(turn_result.answer)
        if not answer_chunks:
            answer_chunks = [""]

        for answer_chunk in answer_chunks:
            yield self._format_sse_payload(
                {
                    "type": "answer_delta",
                    "delta": answer_chunk,
                }
            )

        yield self._format_sse_payload(
            {
                "type": "message_end",
                "session_id": turn_result.session_id,
                "answer": turn_result.answer,
                "model": turn_result.model_name,
                "finish_reason": turn_result.finish_reason,
                "used_tools": turn_result.used_tools,
                "tool_calls": [
                    {
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "output": tool_call.output,
                    }
                    for tool_call in turn_result.tool_calls
                ],
            }
        )

    async def _execute_chat_turn(self, chat_request: ChatRequest) -> ChatTurnResult:
        """执行单轮对话与工具调用流程。"""

        if chat_request.tool_choice and not chat_request.enable_tools:
            raise AppException(
                "未启用工具调用时不能指定 tool_choice。",
                error_code="invalid_request",
            )

        try:
            session_id = await self._ensure_session(
                chat_request.session_id,
                chat_request.user_message,
            )
            await self._message_repository.create(
                message_id=self._generate_identifier(),
                session_id=session_id,
                role="user",
                content=chat_request.user_message,
                message_metadata=chat_request.metadata,
            )

            conversation_messages = self._build_initial_messages(chat_request.user_message)
            available_tools = (
                self._tool_registry.get_tools(chat_request.tool_names or None)
                if chat_request.enable_tools
                else None
            )
            tool_choice = (
                self._tool_registry.normalize_tool_choice(chat_request.tool_choice)
                if chat_request.enable_tools
                else None
            )

            used_tools: list[str] = []
            executed_tool_calls: list[ExecutedToolCall] = []
            final_result: LlmChatCompletionResult | None = None

            for tool_round in range(MAX_TOOL_CALL_ROUNDS):
                completion_result = await self._llm_client.create_chat_completion(
                    messages=conversation_messages,
                    model_name=chat_request.model,
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
                        used_tools.append(tool_result.tool_name)
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
            answer=final_result.content,
            model_name=final_result.model_name,
            prompt_tokens=final_result.prompt_tokens,
            completion_tokens=final_result.completion_tokens,
            total_tokens=final_result.total_tokens,
            finish_reason=final_result.finish_reason,
            used_tools=used_tools,
            tool_calls=executed_tool_calls,
        )

    async def _ensure_session(self, session_id: str | None, user_message: str) -> str:
        """确保当前请求绑定到可用会话。"""

        if session_id is None:
            session_entity = await self._session_repository.create(
                session_id=self._generate_identifier(),
                title=user_message[:20],
                user_id=None,
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
    def _build_initial_messages(user_message: str) -> list[LlmInputMessage]:
        """构建单轮对话的基础提示词消息。"""

        return [
            LlmInputMessage(role="system", content=DEFAULT_SYSTEM_PROMPT),
            LlmInputMessage(role="user", content=user_message),
        ]

    @staticmethod
    def _build_chat_response(turn_result: ChatTurnResult) -> ChatResponse:
        """将内部执行结果转换为 API 响应。"""

        return ChatResponse(
            session_id=turn_result.session_id,
            answer=turn_result.answer,
            model=turn_result.model_name,
            finish_reason=turn_result.finish_reason,
            used_knowledge=False,
            used_tools=turn_result.used_tools,
            tool_calls=[
                ChatToolCallResponse(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    output=tool_call.output,
                )
                for tool_call in turn_result.tool_calls
            ],
        )

    @staticmethod
    def _format_sse_payload(payload: dict[str, object]) -> str:
        """将内部聊天事件转换为 SSE 文本。"""

        return f"data: {dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _split_text_for_stream(content: str, chunk_size: int = 20) -> list[str]:
        """按固定长度切分文本，用于最小流式输出。"""

        if not content:
            return []

        return [content[index : index + chunk_size] for index in range(0, len(content), chunk_size)]

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
