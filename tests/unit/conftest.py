"""单元测试共享夹具。

负责为 repository 单元测试提供轻量异步数据库会话。
当前阶段使用 SQLite 仅用于测试隔离，不改变正式环境使用 PostgreSQL 的目标。
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.persistence.base import Base


@pytest_asyncio.fixture
async def db_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """创建每个测试独立的异步 SQLite 引擎。"""

    database_path = tmp_path / "repository-test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """创建每个测试独立的异步数据库会话。"""

    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session
