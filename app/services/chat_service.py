"""对话服务模块。

负责基础单轮对话的业务编排，包括会话创建、消息落库和答案生成。
当前阶段不负责多轮记忆、LangGraph 状态图和知识库路由决策。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.llm_client import LlmClient
from app.core.exceptions import ResourceNotFoundException
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.chat import ChatRequest, ChatResponse


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


class ChatService:
    """基础单轮对话服务。"""

    def __init__(self, db_session: AsyncSession, llm_client: LlmClient | None = None) -> None:
        self._db_session = db_session
        self._session_repository = SessionRepository(db_session)
        self._message_repository = MessageRepository(db_session)
        self._llm_client = llm_client or LlmClient()

    async def send_message(self, chat_request: ChatRequest) -> ChatResponse:
        """处理单轮对话请求并持久化本轮消息。"""

        turn_result = await self.send_prompt_messages(
            prompt_messages=[
                ("system", "你是最小可用 Agent 后端中的基础问答模块，需要简洁、准确地回答用户。"),
                ("user", chat_request.user_message),
            ],
            latest_user_message=chat_request.user_message,
            session_id=chat_request.session_id,
        )

        return ChatResponse(
            session_id=turn_result.session_id,
            answer=turn_result.answer,
            used_knowledge=False,
            used_tools=[],
        )

    async def send_prompt_messages(
        self,
        *,
        prompt_messages: Sequence[tuple[str, str]],
        latest_user_message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
    ) -> ChatTurnResult:
        """处理标准提示词消息并持久化当前轮次结果。"""

        try:
            if session_id is None:
                session_entity = await self._session_repository.create(
                    session_id=self._generate_identifier(),
                    title=latest_user_message[:20],
                    user_id=user_id,
                )
                session_id = session_entity.session_id
            else:
                session_entity = await self._session_repository.get_by_id(session_id)
                if session_entity is None:
                    raise ResourceNotFoundException(
                        "会话不存在",
                        details={"session_id": session_id},
                    )

            await self._message_repository.create(
                message_id=self._generate_identifier(),
                session_id=session_id,
                role="user",
                content=latest_user_message,
                message_metadata={},
            )

            completion_result = await self._llm_client.create_chat_completion(
                messages=prompt_messages,
                model_name=model_name,
            )

            await self._message_repository.create(
                message_id=self._generate_identifier(),
                session_id=session_id,
                role="assistant",
                content=completion_result.content,
            )
            await self._session_repository.update_timestamp(session_id)
            await self._db_session.commit()
        except Exception:
            await self._db_session.rollback()
            raise

        return ChatTurnResult(
            session_id=session_id,
            answer=completion_result.content,
            model_name=completion_result.model_name,
            prompt_tokens=completion_result.prompt_tokens,
            completion_tokens=completion_result.completion_tokens,
            total_tokens=completion_result.total_tokens,
            finish_reason=completion_result.finish_reason,
        )

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
