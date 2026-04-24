"""Agent 图主链路集成测试。"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.graph import ConversationGraph
from app.agent.state import ChatExecutionRequest, ChatTurnResult, ExecutionPlan, ExecutionStep, ResolvedArguments
from app.clients.llm_client import LlmInputMessage
from app.persistence.base import Base
from app.persistence.memory_repo import MemoryRepository
from app.persistence.message_repo import MessageRepository
from app.persistence.session_repo import SessionRepository
from app.schemas.knowledge import KnowledgeSearchResult


def test_conversation_graph_initial_state_keeps_forced_route() -> None:
    """scheduled_route request value should become a force-route override in graph state."""

    execution_request = ChatExecutionRequest(
        session_id="session-001",
        need_session_memory=False,
        latest_user_message="请提供省内整体实时路况总结。",
        input_messages=[],
        model_name="test-model",
        requested_tool_names=None,
        tool_choice=None,
        scheduled_route="report",
    )

    initial_state = ConversationGraph._build_initial_state(execution_request)

    assert initial_state["forced_route"] == "report"
    assert "scheduled_route" not in initial_state


@pytest.mark.asyncio
async def test_conversation_graph_forced_report_route_overrides_planner_answer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A request-level report override should hit report_node even when the planner says answer."""

    async def fake_planner_run(self, state):
        del self, state
        execution_plan = ExecutionPlan(
            primary_category="general",
            execution_mode="direct",
            recommended_route="answer",
            steps=[
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="直接回答",
                )
            ],
        )
        return {
            "primary_category": execution_plan.primary_category,
            "execution_plan": execution_plan,
            "need_clarification": False,
            "clarification_question": None,
            "steps": execution_plan.steps,
        }

    async def fake_argument_run(self, state):
        del self, state
        return {
            "resolved_arguments": ResolvedArguments(category="general"),
            "step_arguments": {},
            "need_clarification": False,
            "clarification_question": None,
        }

    async def fake_report_run(self, state):
        del self, state
        return {"report_context": "forced report"}

    async def fake_answer_run(self, state):
        del self, state
        return {
            "final_result": ChatTurnResult(
                session_id="session-report-001",
                content="测试模型回答：报表完成",
                model_name="test-model",
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                finish_reason="stop",
                route="answer",
            )
        }

    async def fake_memory_run(self, state):
        del self, state
        return {}

    async def fake_load_checkpoint(self: object, session_id: str) -> dict[str, object] | None:
        del self, session_id
        return None

    async def fake_save_checkpoint(
        self: object,
        *,
        session_id: str,
        payload: dict[str, object],
        ttl_seconds: int = 3600,
    ) -> None:
        del self, session_id, payload, ttl_seconds

    monkeypatch.setattr("app.agent.nodes.planner_node.PlannerNode.run", fake_planner_run)
    monkeypatch.setattr("app.agent.nodes.argument_node.ArgumentNode.run", fake_argument_run)
    monkeypatch.setattr("app.agent.nodes.report_node.ReportNode.run", fake_report_run)
    monkeypatch.setattr("app.agent.nodes.answer_node.AnswerNode.run", fake_answer_run)
    monkeypatch.setattr("app.agent.nodes.memory_node.MemoryNode.run", fake_memory_run)
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.load", fake_load_checkpoint)
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.save", fake_save_checkpoint)

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'agent-graph-forced-report.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        session_repository = SessionRepository(db_session)
        await session_repository.create(session_id="session-report-001")
        conversation_graph = ConversationGraph(db_session)

        execution_request = ChatExecutionRequest(
            session_id="session-report-001",
            need_session_memory=False,
            latest_user_message="你好",
            input_messages=[LlmInputMessage(role="user", content="你好")],
            model_name="test-model",
            requested_tool_names=None,
            tool_choice=None,
            scheduled_route="report",
        )

        started_nodes: list[str] = []
        async for event in conversation_graph.stream_events(execution_request):
            if event["event"] == "on_chain_start" and isinstance(event.get("name"), str):
                started_nodes.append(str(event["name"]))

        report_index = started_nodes.index("report_node")
        answer_index = started_nodes.index("answer_node")

        assert report_index < answer_index

    await engine.dispose()


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
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AIMessage:
        """根据上下文中的历史用户消息生成稳定回答。"""

        del self, model_name, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking
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
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AIMessage:
        """为 planner 和 answer 返回稳定结果。"""

        del self, model_name, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking
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


@pytest.mark.asyncio
async def test_conversation_graph_chains_route_then_traffic_for_od_congestion(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """OD + 拥堵问题应按 route -> traffic -> answer 链路执行。"""

    async def fake_create_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AIMessage:
        del self, model_name, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking
        message_texts = [str(getattr(message, "content", "")) for message in messages]
        if any("生成分类与执行计划" in message for message in message_texts):
            return AIMessage(
                content=(
                    '{"primary_category": "traffic_status", "need_clarification": false, '
                    '"clarification_question": null, "steps": ['
                    '{"step_id": "route_1", "executor": "route", "goal": "查询路线", "depends_on": []}, '
                    '{"step_id": "traffic_1", "executor": "traffic", "goal": "查询路况", "depends_on": ["route_1"]}, '
                    '{"step_id": "answer_1", "executor": "answer", "goal": "总结回答", "depends_on": ["traffic_1"]}'
                    "]}",
                ),
                response_metadata={"finish_reason": "stop"},
                usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            )

        return AIMessage(
            content="测试模型回答：杭州到金华目前有拥堵路段，请注意绕行。",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )

    def fake_create_runnable(self: object, **kwargs: object) -> object:
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

    async def fake_execute_named_tool(
        self: object,
        *,
        tool_name: str,
        arguments: dict[str, object],
    ) -> str:
        del self
        if tool_name == "live_driving_query":
            assert arguments == {"start": "杭州", "end": "金华"}
            return (
                '{"routesCount": 1, "routes": [{"distance": 180000, "duration": 120, '
                '"toll": 85, "sections": [{"roadName": "杭金衢高速", "trafficControls": [], '
                '"serviceAreas": [{"serviceName": "诸暨服务区"}]}]}]}'
            )
        if tool_name == "live_road_event_query":
            assert arguments == {"road": "杭金衢高速"}
            return (
                '[{"roadName": "杭金衢高速", "congestionInfoList": [{"id": "cg-1"}], '
                '"trafficControlList": [], "serviceAreaList": [], "exitInfoList": []}]'
            )
        raise AssertionError(f"unexpected tool: {tool_name}")

    async def fake_load_checkpoint(self: object, session_id: str) -> dict[str, object] | None:
        del self, session_id
        return None

    async def fake_save_checkpoint(
        self: object,
        *,
        session_id: str,
        payload: dict[str, object],
        ttl_seconds: int = 3600,
    ) -> None:
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
        "app.tools.registry.ToolRegistry.execute_named_tool",
        fake_execute_named_tool,
    )
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.load", fake_load_checkpoint)
    monkeypatch.setattr("app.memory.checkpoint_store.CheckpointStore.save", fake_save_checkpoint)

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'agent-graph-route-traffic.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        session_repository = SessionRepository(db_session)
        await session_repository.create(session_id="session-route-traffic-001")
        conversation_graph = ConversationGraph(db_session)

        execution_request = ChatExecutionRequest(
            session_id="session-route-traffic-001",
            need_session_memory=False,
            latest_user_message="杭州到金华堵不堵",
            input_messages=[LlmInputMessage(role="user", content="杭州到金华堵不堵")],
            model_name="test-model",
            requested_tool_names=None,
            tool_choice=None,
        )

        started_nodes: list[str] = []
        async for event in conversation_graph.stream_events(execution_request):
            if event["event"] == "on_chain_start" and isinstance(event.get("name"), str):
                started_nodes.append(str(event["name"]))

        route_index = started_nodes.index("route_node")
        traffic_index = started_nodes.index("traffic_node")
        answer_index = started_nodes.index("answer_node")

        assert route_index < traffic_index < answer_index
        assert started_nodes.count("scheduler_node") >= 3

    await engine.dispose()
