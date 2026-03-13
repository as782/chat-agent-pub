"""会话仓储单元测试。"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.session_repo import SessionRepository


@pytest.mark.asyncio
async def test_session_repository_create_and_get_by_id(db_session: AsyncSession) -> None:
    """验证会话仓储能够创建并查询会话。"""

    repository = SessionRepository(db_session)

    created_session = await repository.create(
        session_id="session-001",
        title="测试会话",
        user_id="user-001",
    )
    queried_session = await repository.get_by_id("session-001")

    assert created_session.session_id == "session-001"
    assert queried_session is not None
    assert queried_session.title == "测试会话"
    assert queried_session.user_id == "user-001"


@pytest.mark.asyncio
async def test_session_repository_list_filters_by_user_and_orders_by_latest(
    db_session: AsyncSession,
) -> None:
    """验证会话列表支持按用户过滤，并按最近更新时间倒序排列。"""

    repository = SessionRepository(db_session)
    base_time = datetime(2026, 3, 13, tzinfo=UTC)

    await repository.create(
        session_id="session-001",
        title="较旧会话",
        user_id="user-001",
        created_at=base_time,
    )
    await repository.create(
        session_id="session-002",
        title="较新会话",
        user_id="user-001",
        created_at=base_time + timedelta(minutes=1),
    )
    await repository.create(
        session_id="session-003",
        title="其他用户会话",
        user_id="user-002",
        created_at=base_time + timedelta(minutes=2),
    )

    session_list = await repository.list(user_id="user-001")

    assert [session.session_id for session in session_list] == ["session-002", "session-001"]
