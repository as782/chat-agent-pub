"""数据库连接模块。

负责创建异步数据库引擎、会话工厂和 FastAPI 依赖。
当前阶段不负责数据库迁移脚本和复杂连接池治理。
"""

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.persistence.base import Base


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """创建并缓存异步数据库引擎。"""

    settings = get_settings()
    # 数据库 SQL 细节默认不直接输出到控制台，避免本地联调被大量底层日志淹没。
    return create_async_engine(settings.database_url, echo=False)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """创建并缓存异步会话工厂。"""

    return async_sessionmaker(bind=get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """提供 FastAPI 请求级数据库会话。"""

    session_factory = get_session_factory()
    async with session_factory() as db_session:
        yield db_session


async def initialize_database() -> None:
    """初始化数据库表结构。

    这里显式导入模型是为了确保所有 ORM 表都已注册到 metadata。
    """

    from app.persistence import models as persistence_models  # noqa: F401

    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_database() -> None:
    """释放数据库连接并清理缓存。"""

    engine = get_engine()
    await engine.dispose()
    clear_database_caches()


def clear_database_caches() -> None:
    """清理数据库相关缓存，供测试场景复用。"""

    get_session_factory.cache_clear()
    get_engine.cache_clear()
