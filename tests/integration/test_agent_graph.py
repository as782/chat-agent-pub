"""Agent 图主链路集成测试。"""

from pathlib import Path

import pytest
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.graph import ConversationGraph
from app.agent.state import ChatExecutionRequest
from langchain_core.messages import AIMessage
from app.clients.llm_client import LlmInputMessage
from app.persistence.base import Base
from app.persistence.memory_repo import MemoryRepository
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository


@pytest.mark.asyncio
async def test_conversation_graph_reuses_session_history(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """验证 LangGraph 主链路会自动复用同一会话下的历史消息。"""

    async def fake_create_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AIMessage:
        """根据上下文中的历史用户消息生成稳定回答。"""

        del self, tools, tool_choice, enable_thinking
        user_messages = [
            str(getattr(message, "content", ""))
            for message in messages
            if getattr(message, "role", "") == "user"
        ]
        latest_user_message = user_messages[-1] if user_messages else ""

        if "我刚刚告诉你的名字是什么" in latest_user_message and any(
            "我叫小王" in message for message in user_messages[:-1]
        ):
            return AIMessage(
                content="测试模型回答：你刚刚说你叫小王",
                response_metadata={"finish_reason": "stop"},
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

        return AIMessage(
            content=f"测试模型回答：{latest_user_message}",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )

    async def fake_load_checkpoint(self: object, session_id: str) -> dict[str, object] | None:
        """测试场景下不依赖真实 Redis。"""

        del self, session_id
        return None

    async def fake_save_checkpoint(
        self: object,
        *,
        session_id: str,
        payload: dict[str, object],
        ttl_seconds: int = 3600,
    ) -> None:
        """测试场景下跳过 Redis 写入。"""

        del self, session_id, payload, ttl_seconds

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.load", fake_load_checkpoint)
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.save", fake_save_checkpoint)

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'agent-graph.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        await _prepare_session_history(db_session)
        conversation_graph = ConversationGraph(db_session)

        execution_request = ChatExecutionRequest(
            session_id="session-001",
            need_session_memory=True,
            latest_user_message="我刚刚告诉你的名字是什么？",
            input_messages=[
                LlmInputMessage(role="user", content="我刚刚告诉你的名字是什么？"),
            ],
            model_name="test-model",
            requested_tool_names=None,
            tool_choice=None,
        )
        turn_result, checkpoint_payload = await conversation_graph.run_turn(execution_request)
        memory_entity = await MemoryRepository(db_session).get_by_session_id("session-001")

        assert turn_result.content == "测试模型回答：你刚刚说你叫小王"
        assert turn_result.used_session_memory is True
        assert checkpoint_payload is not None
        assert memory_entity is not None
        assert memory_entity.message_count >= 3

    await engine.dispose()


async def _prepare_session_history(db_session: AsyncSession) -> None:
    """准备图测试所需的基础会话和已持久化用户消息。"""

    session_repository = SessionRepository(db_session)
    message_repository = MessageRepository(db_session)
    await session_repository.create(session_id="session-001")
    await message_repository.create(
        message_id="message-001",
        session_id="session-001",
        role="user",
        content="我叫小王，请记住这个名字。",
    )
    await message_repository.create(
        message_id="message-002",
        session_id="session-001",
        role="assistant",
        content="好的，我记住了。",
    )
    await message_repository.create(
        message_id="message-003",
        session_id="session-001",
        role="user",
        content="我刚刚告诉你的名字是什么？",
    )
