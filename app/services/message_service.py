"""消息服务模块。

负责消息历史查询等业务编排，不承担消息落库存储实现。
当前阶段不负责消息分页策略优化和多轮上下文裁剪。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundException
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.message import MessageListResponse, MessageResponse


class MessageService:
    """消息历史服务。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session
        self._session_repository = SessionRepository(db_session)
        self._message_repository = MessageRepository(db_session)

    async def list_messages(
        self,
        *,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> MessageListResponse:
        """查询指定会话的消息历史。"""

        session_entity = await self._session_repository.get_by_id(session_id)
        if session_entity is None:
            raise ResourceNotFoundException(
                "会话不存在",
                details={"session_id": session_id},
            )

        message_entities = await self._message_repository.list_by_session(
            session_id,
            limit=limit,
            offset=offset,
        )
        total = await self._message_repository.count_by_session(session_id)
        return MessageListResponse(
            items=[
                MessageResponse(
                    message_id=message_entity.message_id,
                    session_id=message_entity.session_id,
                    role=message_entity.role,
                    content=message_entity.content,
                    metadata=message_entity.message_metadata,
                    created_at=message_entity.created_at,
                )
                for message_entity in message_entities
            ],
            total=total,
        )
