"""记忆节点模块。
负责在回答生成后刷新短期记忆快照，并准备 checkpoint 负载。
当前阶段不负责长期记忆归档和复杂冲突合并。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import AgentState
from app.memory.manager import MemoryManager


class MemoryNode:
    """LangGraph 记忆节点。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._memory_manager = memory_manager or MemoryManager(db_session)

    async def run(self, state: AgentState) -> dict[str, object]:
        """刷新当前会话的短期记忆。"""

        checkpoint_payload = await self.refresh_session_memory(
            session_id=str(state["session_id"]),
            route=str(state.get("route", "answer")),
        )
        return {"checkpoint_payload": checkpoint_payload}

    async def refresh_session_memory(
        self,
        *,
        session_id: str,
        route: str,
    ) -> dict[str, object]:
        """对外暴露刷新记忆能力，供流式路径复用。"""

        return await self._memory_manager.refresh_memory(session_id=session_id, route=route)

    async def save_checkpoint(self, checkpoint_payload: dict[str, object] | None) -> None:
        """在事务提交后保存 checkpoint。"""

        if checkpoint_payload is None:
            return
        await self._memory_manager.save_checkpoint(checkpoint_payload)
