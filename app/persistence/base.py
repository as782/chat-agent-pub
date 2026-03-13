"""持久化基础模块。

负责定义 SQLAlchemy 的声明式基类与通用时间工具。
当前阶段不负责数据库引擎创建、迁移脚本和事务管理。
"""

from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """所有 ORM 模型共享的声明式基类。"""


def get_utc_now() -> datetime:
    """返回统一的 UTC 时间，避免各仓储自行处理时区。"""

    return datetime.now(UTC)
