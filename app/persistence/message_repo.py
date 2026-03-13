"""消息仓储模块。

负责消息表的数据访问，不承担消息裁剪、上下文拼装等业务策略。
当前阶段不负责事务提交，由上层调用方统一控制事务边界。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Select, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.base import get_utc_now
from app.persistence.models import MessageEntity


class MessageRepository:
    """消息数据访问仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    async def create(
        self,
        *,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        message_metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> MessageEntity:
        """创建消息记录。"""

        message_entity = MessageEntity(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            message_metadata=message_metadata or {},
            created_at=created_at or get_utc_now(),
        )
        self._db_session.add(message_entity)
        await self._db_session.flush()
        await self._db_session.refresh(message_entity)
        return message_entity

    async def list_by_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MessageEntity]:
        """按会话分页查询消息历史。"""

        statement: Select[tuple[MessageEntity]] = (
            select(MessageEntity)
            .where(MessageEntity.session_id == session_id)
            .order_by(MessageEntity.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._db_session.execute(statement)
        return list(result.scalars().all())

    async def get_latest_by_session(self, session_id: str) -> MessageEntity | None:
        """查询指定会话的最新一条消息。"""

        statement: Select[tuple[MessageEntity]] = (
            select(MessageEntity)
            .where(MessageEntity.session_id == session_id)
            .order_by(MessageEntity.created_at.desc())
            .limit(1)
        )
        result = await self._db_session.execute(statement)
        return result.scalars().first()

    async def delete_by_session(self, session_id: str) -> int:
        """删除指定会话下的全部消息。"""

        statement = delete(MessageEntity).where(MessageEntity.session_id == session_id)
        result = await self._db_session.execute(statement)
        await self._db_session.flush()
        return int(result.rowcount or 0)
