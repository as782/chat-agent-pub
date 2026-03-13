"""对话服务模块。

负责基础单轮对话的业务编排，包括会话创建、消息落库和答案生成。
当前阶段不负责多轮记忆、LangGraph 状态图和知识库路由决策。
"""

from uuid import uuid4

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundException
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.chat import ChatRequest, ChatResponse


class ChatService:
    """基础单轮对话服务。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session
        self._session_repository = SessionRepository(db_session)
        self._message_repository = MessageRepository(db_session)
        self._answer_chain = ChatPromptTemplate.from_messages(
            [
                ("system", "你是最小可用 Agent 后端中的基础问答模块，需要简洁回答用户。"),
                ("human", "{user_message}"),
            ]
        ) | RunnableLambda(self._build_single_turn_answer)

    def _build_single_turn_answer(self, prompt_value: ChatPromptValue) -> str:
        """基于 LangChain 提示词构造基础单轮回答。"""

        user_message = str(prompt_value.messages[-1].content)
        return f"当前为基础单轮模式，我已收到你的问题：{user_message}"

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

            answer_text = self._answer_chain.invoke({"user_message": chat_request.user_message})

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
