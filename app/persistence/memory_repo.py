"""短期记忆仓储模块。

负责短期记忆快照的数据访问，不承担记忆摘要生成和上下文压缩策略。
当前阶段不负责事务提交，由上层调用方统一控制事务边界。
"""

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.base import get_utc_now
from app.persistence.models import MemoryEntity


class MemoryRepository:
    """短期记忆数据访问仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    async def upsert(
        self,
        *,
        session_id: str,
        summary: str | None,
        context_window: dict[str, Any] | None = None,
        message_count: int = 0,
        updated_at: datetime | None = None,
    ) -> MemoryEntity:
        """创建或更新短期记忆快照。"""

        memory_entity = await self.get_by_session_id(session_id)
        if memory_entity is None:
            memory_entity = MemoryEntity(
                session_id=session_id,
                summary=summary,
                context_window=context_window or {},
                message_count=message_count,
                updated_at=updated_at or get_utc_now(),
            )
            self._db_session.add(memory_entity)
        else:
            memory_entity.summary = summary
            memory_entity.context_window = context_window or {}
            memory_entity.message_count = message_count
            memory_entity.updated_at = updated_at or get_utc_now()

        await self._db_session.flush()
        await self._db_session.refresh(memory_entity)
        return memory_entity

    async def get_by_session_id(self, session_id: str) -> MemoryEntity | None:
        """按会话标识查询短期记忆快照。"""

        return await self._db_session.get(MemoryEntity, session_id)

    async def delete(self, session_id: str) -> bool:
        """删除指定会话的短期记忆快照。"""

        memory_entity = await self.get_by_session_id(session_id)
        if memory_entity is None:
            return False

        await self._db_session.delete(memory_entity)
        await self._db_session.flush()
        return True
