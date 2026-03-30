"""Agent 图主链路集成测试。"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.graph import ConversationGraph
from app.agent.state import ChatExecutionRequest
from app.clients.llm_client import LlmInputMessage
from app.persistence.base import Base
from app.persistence.memory_repo import MemoryRepository
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.knowledge import KnowledgeSearchResult


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

        del self, model_name, tools, tool_choice, enable_thinking
        message_texts = [str(getattr(message, "content", "")) for message in messages]
        if any("生成分类与执行计划" in message for message in message_texts):
            return AIMessage(
                content=(
                    '{"primary_category": "general", "need_clarification": false, '
                    '"clarification_question": null, "steps": ['
                    '{"step_id": "answer_1", "executor": "answer", "goal": "直接回答"}]}'
                ),
                response_metadata={"finish_reason": "stop"},
                usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            )

        user_messages = []
        for message in messages:
            message_role = getattr(message, "role", None) or getattr(message, "type", "")
            if str(message_role) in {"user", "human"}:
                user_messages.append(str(getattr(message, "content", "")))
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

    def fake_create_runnable(self: object, **kwargs: object) -> object:
        """让 AnswerNode 通过 astream 获取稳定的测试输出。"""

        del self, kwargs

        class _FakeRunnable:
            async def astream(self, messages: list[object], config=None):
                del config
                ai_message = await fake_create_chat_completion(object(), messages)
                yield AIMessageChunk(
                    content=ai_message.content,
                    response_metadata=ai_message.response_metadata or {},
                    usage_metadata=ai_message.usage_metadata or {},
                    tool_call_chunks=[],
                )

        return _FakeRunnable()

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
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fake_create_runnable,
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


@pytest.mark.asyncio
async def test_conversation_graph_routes_ragflow_directly_to_answer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """验证知识库分支执行后会直接收束到 answer_node。"""

    async def fake_create_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AIMessage:
        """为 planner 和 answer 返回稳定结果。"""

        del self, model_name, tools, tool_choice, enable_thinking
        message_texts = [str(getattr(message, "content", "")) for message in messages]
        if any("生成分类与执行计划" in message for message in message_texts):
            return AIMessage(
                content=(
                    '{"primary_category": "policy", "need_clarification": false, '
                    '"clarification_question": null, "steps": ['
                    '{"step_id": "rag_1", "executor": "rag", "goal": "检索知识"}, '
                    '{"step_id": "answer_1", "executor": "answer", "goal": "汇总回答", '
                    '"depends_on": ["rag_1"]}]}'
                ),
                response_metadata={"finish_reason": "stop"},
                usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            )

        return AIMessage(
            content="测试模型回答：根据知识库，西湖位于杭州。",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )

    def fake_create_runnable(self: object, **kwargs: object) -> object:
        """让 AnswerNode 通过 astream 获取稳定的测试输出。"""

        del self, kwargs

        class _FakeRunnable:
            async def astream(self, messages: list[object], config=None):
                del config
                ai_message = await fake_create_chat_completion(object(), messages)
                yield AIMessageChunk(
                    content=ai_message.content,
                    response_metadata=ai_message.response_metadata or {},
                    usage_metadata=ai_message.usage_metadata or {},
                    tool_call_chunks=[],
                )

        return _FakeRunnable()

    async def fake_retrieve_for_agent(
        self: object,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        """返回稳定的知识检索结果。"""

        del self, top_k
        assert query == "西湖在哪里？"
        return [
            KnowledgeSearchResult(
                document_id="doc-001",
                chunk_id="chunk-001",
                score=0.98,
                content="西湖位于杭州。",
                source="杭州百科",
            )
        ]

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
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fake_create_runnable,
    )
    monkeypatch.setattr(
        "app.agent.nodes.ragflow_node.KnowledgeService.retrieve_for_agent",
        fake_retrieve_for_agent,
    )
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.load", fake_load_checkpoint)
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.save", fake_save_checkpoint)

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'agent-graph-rag.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        session_repository = SessionRepository(db_session)
        await session_repository.create(session_id="session-rag-001")
        conversation_graph = ConversationGraph(db_session)

        execution_request = ChatExecutionRequest(
            session_id="session-rag-001",
            need_session_memory=False,
            latest_user_message="知识库: 西湖在哪里？",
            input_messages=[LlmInputMessage(role="user", content="知识库: 西湖在哪里？")],
            model_name="test-model",
            requested_tool_names=None,
            tool_choice=None,
        )

        started_nodes: list[str] = []
        async for event in conversation_graph.stream_events(execution_request):
            if event["event"] == "on_chain_start" and isinstance(event.get("name"), str):
                started_nodes.append(str(event["name"]))

        ragflow_index = started_nodes.index("ragflow_node")
        assert started_nodes[ragflow_index + 1] == "answer_node"
        assert started_nodes.count("scheduler_node") == 1

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
