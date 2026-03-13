"""短期记忆仓储单元测试。"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.memory_repo import MemoryRepository
from app.persistence.session_repo import SessionRepository


@pytest.mark.asyncio
async def test_memory_repository_upsert_creates_new_record(db_session: AsyncSession) -> None:
    """验证首次写入会创建短期记忆记录。"""

    session_repository = SessionRepository(db_session)
    memory_repository = MemoryRepository(db_session)

    await session_repository.create(session_id="session-001")
    memory = await memory_repository.upsert(
        session_id="session-001",
        summary="第一次摘要",
        context_window={"latest_user_message": "你好"},
        message_count=1,
    )

    assert memory.summary == "第一次摘要"
    assert memory.context_window == {"latest_user_message": "你好"}
    assert memory.message_count == 1


@pytest.mark.asyncio
async def test_memory_repository_upsert_updates_existing_record(db_session: AsyncSession) -> None:
    """验证重复写入会更新已有短期记忆记录。"""

    session_repository = SessionRepository(db_session)
    memory_repository = MemoryRepository(db_session)

    await session_repository.create(session_id="session-001")
    await memory_repository.upsert(
        session_id="session-001",
        summary="旧摘要",
        context_window={"latest_user_message": "旧内容"},
        message_count=1,
    )

    memory = await memory_repository.upsert(
        session_id="session-001",
        summary="新摘要",
        context_window={"latest_user_message": "新内容"},
        message_count=3,
    )

    assert memory.summary == "新摘要"
    assert memory.context_window == {"latest_user_message": "新内容"}
    assert memory.message_count == 3
