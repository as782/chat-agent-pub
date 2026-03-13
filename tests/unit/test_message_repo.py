"""消息仓储单元测试。"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository


@pytest.mark.asyncio
async def test_message_repository_lists_messages_in_created_order(db_session: AsyncSession) -> None:
    """验证消息仓储按创建时间升序返回消息。"""

    session_repository = SessionRepository(db_session)
    message_repository = MessageRepository(db_session)
    base_time = datetime(2026, 3, 13, tzinfo=UTC)

    await session_repository.create(session_id="session-001")
    await message_repository.create(
        message_id="message-001",
        session_id="session-001",
        role="user",
        content="第一条消息",
        created_at=base_time,
    )
    await message_repository.create(
        message_id="message-002",
        session_id="session-001",
        role="assistant",
        content="第二条消息",
        created_at=base_time + timedelta(seconds=1),
    )

    message_list = await message_repository.list_by_session("session-001")
    latest_message = await message_repository.get_latest_by_session("session-001")

    assert [message.message_id for message in message_list] == ["message-001", "message-002"]
    assert latest_message is not None
    assert latest_message.message_id == "message-002"


@pytest.mark.asyncio
async def test_message_repository_delete_by_session_returns_deleted_count(
    db_session: AsyncSession,
) -> None:
    """验证删除会话消息时会返回实际删除数量。"""

    session_repository = SessionRepository(db_session)
    message_repository = MessageRepository(db_session)

    await session_repository.create(session_id="session-001")
    await message_repository.create(
        message_id="message-001",
        session_id="session-001",
        role="user",
        content="需要删除的消息",
    )

    deleted_count = await message_repository.delete_by_session("session-001")
    remaining_messages = await message_repository.list_by_session("session-001")

    assert deleted_count == 1
    assert remaining_messages == []
