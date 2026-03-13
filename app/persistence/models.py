"""持久化模型定义。

负责定义会话、消息、短期记忆和 RAGFlow 数据集映射的 ORM 模型。
当前阶段不负责数据库迁移版本管理和复杂索引调优。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.base import Base, get_utc_now


class SessionEntity(Base):
    """会话表模型。"""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now)

    messages: Mapped[list["MessageEntity"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class MessageEntity(Base):
    """消息表模型。"""

    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text())
    message_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, index=True
    )

    session: Mapped[SessionEntity] = relationship(back_populates="messages")


class MemoryEntity(Base):
    """短期记忆表模型。"""

    __tablename__ = "memories"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    summary: Mapped[str | None] = mapped_column(Text(), nullable=True)
    context_window: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now)


class RagflowDatasetEntity(Base):
    """RAGFlow 数据集映射表模型。"""

    __tablename__ = "ragflow_datasets"

    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(255))
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    dataset_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now)
