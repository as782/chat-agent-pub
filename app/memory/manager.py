"""记忆管理模块。
负责协调消息历史、摘要器、仓储和 checkpoint 存储，形成最小可用短期记忆能力。
当前阶段不负责长期记忆召回和复杂冲突消解。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context_builder import deserialize_input_messages, serialize_input_messages
from app.clients.llm_client import LlmInputMessage
from app.memory.checkpoint_store import CheckpointStore
from app.memory.summarizer import MemorySummarizer
from app.persistence.memory_repo import MemoryRepository
from app.persistence.message_repo import MessageRepository


@dataclass(slots=True)
class SessionMemorySnapshot:
    """当前会话可用的记忆快照。"""

    summary: str | None
    context_window_messages: list[LlmInputMessage]
    message_count: int
    checkpoint_payload: dict[str, Any] | None = None


class MemoryManager:
    """短期记忆管理器。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        checkpoint_store: CheckpointStore | None = None,
        summarizer: MemorySummarizer | None = None,
    ) -> None:
        self._message_repository = MessageRepository(db_session)
        self._memory_repository = MemoryRepository(db_session)
        self._checkpoint_store = checkpoint_store or CheckpointStore()
        self._summarizer = summarizer or MemorySummarizer()

    async def load_snapshot(self, session_id: str) -> SessionMemorySnapshot:
        """加载当前会话的记忆快照。"""

        memory_entity = await self._memory_repository.get_by_session_id(session_id)
        checkpoint_payload = await self._checkpoint_store.load(session_id)

        if memory_entity is not None:
            context_window = memory_entity.context_window or {}
            messages_payload = context_window.get("messages", [])
            fallback_summary = self._extract_summary_from_checkpoint(checkpoint_payload)
            return SessionMemorySnapshot(
                summary=memory_entity.summary or fallback_summary,
                context_window_messages=deserialize_input_messages(
                    messages_payload if isinstance(messages_payload, list) else []
                ),
                message_count=memory_entity.message_count,
                checkpoint_payload=checkpoint_payload,
            )

        if isinstance(checkpoint_payload, dict):
            messages_payload = checkpoint_payload.get("context_window_messages", [])
            return SessionMemorySnapshot(
                summary=self._extract_summary_from_checkpoint(checkpoint_payload),
                context_window_messages=deserialize_input_messages(
                    messages_payload if isinstance(messages_payload, list) else []
                ),
                message_count=int(checkpoint_payload.get("message_count", 0)),
                checkpoint_payload=checkpoint_payload,
            )

        return SessionMemorySnapshot(
            summary=None,
            context_window_messages=[],
            message_count=0,
            checkpoint_payload=None,
        )

    async def refresh_memory(
        self,
        *,
        session_id: str,
        route: str,
    ) -> dict[str, Any]:
        """根据最新会话消息刷新短期记忆快照。"""

        message_count = await self._message_repository.count_by_session(session_id)
        query_limit = min(max(message_count, 1), 100)
        query_offset = max(message_count - query_limit, 0)
        session_messages = await self._message_repository.list_by_session(
            session_id,
            limit=query_limit,
            offset=query_offset,
        )
        summary_result = self._summarizer.summarize_messages(session_messages)
        serialized_context_window = serialize_input_messages(summary_result.context_window_messages)

        await self._memory_repository.upsert(
            session_id=session_id,
            summary=summary_result.summary,
            context_window={"messages": serialized_context_window},
            message_count=message_count,
        )

        return {
            "session_id": session_id,
            "route": route,
            "summary": summary_result.summary,
            "message_count": message_count,
            "context_window_messages": serialized_context_window,
        }

    async def save_checkpoint(self, checkpoint_payload: dict[str, Any]) -> None:
        """把刷新后的记忆快照保存到 checkpoint 存储。"""

        await self._checkpoint_store.save(
            session_id=str(checkpoint_payload["session_id"]),
            payload=checkpoint_payload,
        )

    @staticmethod
    def _extract_summary_from_checkpoint(
        checkpoint_payload: dict[str, Any] | None,
    ) -> str | None:
        """从 checkpoint 负载中提取摘要文本。"""

        if not isinstance(checkpoint_payload, dict):
            return None
        summary = checkpoint_payload.get("summary")
        return str(summary) if isinstance(summary, str) and summary else None
