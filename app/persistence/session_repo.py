"""会话仓储模块。

负责会话表的数据访问，不承担会话业务决策和状态机判断。
当前阶段不负责事务提交，由上层调用方统一控制事务边界。
"""

from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.base import get_utc_now
from app.persistence.models import SessionEntity


class SessionRepository:
    """会话数据访问仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self._db_session = db_session

    async def create(
        self,
        *,
        session_id: str,
        title: str | None = None,
        user_id: str | None = None,
        status: str = "active",
        created_at: datetime | None = None,
    ) -> SessionEntity:
        """创建会话记录。"""

        current_time = created_at or get_utc_now()
        session_entity = SessionEntity(
            session_id=session_id,
            title=title,
            user_id=user_id,
            status=status,
            created_at=current_time,
            updated_at=current_time,
        )
        self._db_session.add(session_entity)
        await self._db_session.flush()
        await self._db_session.refresh(session_entity)
        return session_entity

    async def get_by_id(self, session_id: str) -> SessionEntity | None:
        """按会话标识查询单条记录。"""

        return await self._db_session.get(SessionEntity, session_id)

    async def list(
        self,
        *,
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SessionEntity]:
        """分页查询会话列表。"""

        statement: Select[tuple[SessionEntity]] = (
            select(SessionEntity)
            .order_by(SessionEntity.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if user_id is not None:
            statement = statement.where(SessionEntity.user_id == user_id)

        result = await self._db_session.execute(statement)
        return list(result.scalars().all())

    async def count(self, *, user_id: str | None = None) -> int:
        """统计会话数量。"""

        statement = select(func.count()).select_from(SessionEntity)
        if user_id is not None:
            statement = statement.where(SessionEntity.user_id == user_id)

        result = await self._db_session.execute(statement)
        return int(result.scalar_one())

    async def update_timestamp(
        self,
        session_id: str,
        *,
        updated_at: datetime | None = None,
    ) -> SessionEntity | None:
        """更新时间戳，用于标记最近活跃时间。"""

        session_entity = await self.get_by_id(session_id)
        if session_entity is None:
            return None

        session_entity.updated_at = updated_at or get_utc_now()
        await self._db_session.flush()
        await self._db_session.refresh(session_entity)
        return session_entity
