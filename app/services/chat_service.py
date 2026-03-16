"""对话服务模块。

负责基础单轮对话的业务编排，包括会话创建、消息落库和答案生成。
当前阶段不负责多轮记忆、LangGraph 状态图和知识库路由决策。
"""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.llm_client import LlmClient
from app.core.exceptions import ResourceNotFoundException
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.chat import ChatRequest, ChatResponse


class ChatService:
    """基础单轮对话服务。"""

    def __init__(self, db_session: AsyncSession, llm_client: LlmClient | None = None) -> None:
        self._db_session = db_session
        self._session_repository = SessionRepository(db_session)
        self._message_repository = MessageRepository(db_session)
        self._llm_client = llm_client or LlmClient()

    async def send_message(self, chat_request: ChatRequest) -> ChatResponse:
        """处理单轮对话请求并持久化本轮消息。"""

        session_id = chat_request.session_id

        try:
            if session_id is None:
                session_entity = await self._session_repository.create(
                    session_id=self._generate_identifier(),
                    title=chat_request.user_message[:20],
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
                content=chat_request.user_message,
                message_metadata=chat_request.metadata,
            )

            answer_text = await self._llm_client.generate_answer(chat_request.user_message)

            await self._message_repository.create(
                message_id=self._generate_identifier(),
                session_id=session_id,
                role="assistant",
                content=answer_text,
            )
            await self._session_repository.update_timestamp(session_id)
            await self._db_session.commit()
        except Exception:
            await self._db_session.rollback()
            raise

        return ChatResponse(
            session_id=session_id,
            answer=answer_text,
            used_knowledge=False,
            used_tools=[],
        )

    @staticmethod
    def _generate_identifier() -> str:
        """生成统一长度的业务标识。"""

        return uuid4().hex
