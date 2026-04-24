"""AnswerNode network report summary tests."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.clients.llm_client import LlmInputMessage
from app.agent.nodes.answer_node import AnswerNode
from app.agent.state import (
    ExecutionPlan,
    ExecutionStep,
    ExecutorResult,
    PreparedContext,
    ResolvedArguments,
)
from app.persistence.base import Base


@pytest.mark.asyncio
async def test_answer_node_uses_llm_for_report_summary_and_keeps_table_stable(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured_messages: dict[str, object] = {}

    async def fake_create_chat_completion(self: object, **kwargs: object) -> AIMessage:
        del self
        captured_messages.update(kwargs)
        return AIMessage(
            content="当前全网以局部管控为主，甬金高速金华段需重点关注。",
            response_metadata={"finish_reason": "stop", "model_name": "test-model"},
            usage_metadata={"input_tokens": 21, "output_tokens": 12, "total_tokens": 33},
        )

    def fail_create_runnable(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("network_report summary path should not enter generic streaming LLM flow")

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fail_create_runnable,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-report-summary.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-report-summary-001",
                "latest_user_message": "请提供省内整体实时路况总结。",
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                    report_context=(
                        "以下是完整 report_content\n"
                        "查询时间：2026-03-31 09:00:00\n"
                        "主线管制：G1512 宁波方向单向封道\n"
                        "收费站管制：佛堂收费站、徐村收费站"
                    ),
                ),
                "execution_plan": ExecutionPlan(
                    primary_category="network_report",
                    execution_mode="single_step",
                    recommended_route="report",
                ),
                "step_results": {
                    "report_1": ExecutorResult(
                        step_id="report_1",
                        executor="report",
                        is_success=True,
                        normalized_result={
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [
                                {
                                    "roadId": "33171",
                                    "roadGBCode": "G1512",
                                    "roadName": "G1512甬金金华段",
                                    "direction": 100701,
                                    "directionName": "宁波方向",
                                    "controlType": 10102,
                                    "controlTypeName": "单向封道",
                                    "des": "因协助地方双江湖湖底隧道施工需要，甬向佛堂分流开始",
                                }
                            ],
                            "exit_top_items": [
                                {
                                    "roadId": "33171",
                                    "roadName": "G1512甬金高速（金华段）",
                                    "direction": 100701,
                                    "directionName": "宁波方向",
                                    "tollName": "佛堂收费站",
                                    "entrance": 1,
                                    "controlType": 10202,
                                    "controlTypeName": "关闭",
                                },
                                {
                                    "roadId": "33171",
                                    "roadName": "G1512甬金高速（金华段）",
                                    "direction": 100701,
                                    "directionName": "宁波方向",
                                    "tollName": "佛堂收费站",
                                    "entrance": 0,
                                    "entranceName": "出口",
                                    "controlType": 10204,
                                    "controlTypeName": "分流",
                                },
                                {
                                    "roadId": "33171",
                                    "roadName": "G1512甬金高速（金华段）",
                                    "direction": 100700,
                                    "directionName": "金华方向",
                                    "tollName": "徐村收费站",
                                    "entrance": 1,
                                    "controlType": 10202,
                                    "controlTypeName": "关闭",
                                },
                                {
                                    "roadId": "33171",
                                    "roadName": "G1512甬金高速（金华段）",
                                    "direction": 100700,
                                    "directionName": "金华方向",
                                    "tollName": "徐村收费站",
                                    "entrance": 0,
                                    "entranceName": "出口",
                                    "controlType": 10204,
                                    "controlTypeName": "分流",
                                },
                            ],
                        },
                    )
                },
            }
        )

        llm_messages = captured_messages["messages"]
        assert isinstance(llm_messages, list)
        assert len(llm_messages) == 3
        assert "report_content" in llm_messages[0].content
        assert "主线管制" in llm_messages[1].content
        assert result["final_result"].content.startswith(
            "当前全网以局部管控为主，甬金高速金华段需重点关注。"
        )
        assert (
            "| G1512 | 甬金高速 | 金华段 | 佛堂收费站，宁波方向入口关闭、出口分流 | 无 |"
            in result["final_result"].content
        )
        assert (
            "| G1512 | 甬金高速 | 金华段 | 徐村收费站，金华方向入口关闭、出口分流 | 无 |"
            in result["final_result"].content
        )
        assert (
            "| G1512 | 甬金高速 | 金华段 | 无 | 宁波方向，单向封道，因协助地方双江湖湖底隧道施工需要 |"
            in result["final_result"].content
        )
        assert result["final_result"].model_name == "test-model"
        assert result["final_result"].total_tokens == 33

    await engine.dispose()


@pytest.mark.asyncio
async def test_answer_node_ignores_placeholder_report_reference_answer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured_messages: dict[str, object] = {}

    async def fake_create_chat_completion(self: object, **kwargs: object) -> AIMessage:
        del self
        captured_messages.update(kwargs)
        return AIMessage(
            content="当前路网整体平稳。",
            response_metadata={"finish_reason": "stop", "model_name": "test-model"},
            usage_metadata={"input_tokens": 21, "output_tokens": 12, "total_tokens": 33},
        )

    def fail_create_runnable(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("placeholder reference_answer should not disable network report rendering")

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fail_create_runnable,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-report-placeholder.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-report-placeholder-001",
                "latest_user_message": "请提供省内整体实时路况总结。",
                "resolved_arguments": ResolvedArguments(
                    category="network_report",
                    arguments={"reference_answer": "无"},
                ),
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                    report_context="完整 report_content",
                ),
                "execution_plan": ExecutionPlan(
                    primary_category="network_report",
                    execution_mode="single_step",
                    recommended_route="report",
                ),
                "step_results": {
                    "report_1": ExecutorResult(
                        step_id="report_1",
                        executor="report",
                        is_success=True,
                        normalized_result={
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [
                                {
                                    "roadGBCode": "G1512",
                                    "roadName": "G1512甬金金华段",
                                    "direction": 100701,
                                    "directionName": "宁波方向",
                                    "controlType": 10102,
                                    "controlTypeName": "单向封道",
                                    "des": "因协助地方双江湖湖底隧道施工需要，甬向佛堂分流开始",
                                }
                            ],
                            "exit_top_items": [],
                        },
                    )
                },
            }
        )

        llm_messages = captured_messages["messages"]
        assert isinstance(llm_messages, list)
        assert len(llm_messages) == 3
        assert "路网播报总结助手" in llm_messages[0].content
        assert "| roadCode | highwayName | roadSection | controls | traffic |" in result[
            "final_result"
        ].content
        assert "G1512" in result["final_result"].content

    await engine.dispose()


@pytest.mark.asyncio
async def test_answer_node_keeps_renderer_even_with_reference_answer_from_history(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured_messages: dict[str, object] = {}

    async def fake_create_chat_completion(self: object, **kwargs: object) -> AIMessage:
        del self
        captured_messages.update(kwargs)
        return AIMessage(
            content="当前路网整体平稳。",
            response_metadata={"finish_reason": "stop", "model_name": "test-model"},
            usage_metadata={"input_tokens": 21, "output_tokens": 12, "total_tokens": 33},
        )

    def fail_create_runnable(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("report summary route should not fall back to generic streaming LLM flow")

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fail_create_runnable,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-report-force-render.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-report-force-render-001",
                "latest_user_message": "请提供省内整体实时路况总结。",
                "step_arguments": {
                    "1": ResolvedArguments(
                        category="network_report",
                        arguments={
                            "query": "省内整体实时路况总结",
                            "reference_answer": "上次报告显示杭州北向拥堵指数为 2.1。",
                        },
                    )
                },
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                    report_context="完整 report_content",
                ),
                "execution_plan": ExecutionPlan(
                    primary_category="network_report",
                    execution_mode="single_step",
                    recommended_route="report",
                    steps=[
                        ExecutionStep(
                            step_id="1",
                            executor="report",
                            goal="获取省内整体实时路况汇总数据",
                            metadata={
                                "query": "省内整体实时路况总结",
                                "scope": "全省",
                                "reference_answer": "上次报告显示杭州北向拥堵指数为 2.1。",
                            },
                        ),
                        ExecutionStep(
                            step_id="2",
                            executor="answer",
                            goal="生成最终回答",
                            depends_on=["1"],
                        ),
                    ],
                ),
                "step_results": {
                    "1": ExecutorResult(
                        step_id="1",
                        executor="report",
                        is_success=True,
                        normalized_result={
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [],
                            "exit_top_items": [
                                {
                                    "roadGBCode": "G1512",
                                    "roadName": "G1512甬金高速（金华段）",
                                    "directionName": "宁波方向",
                                    "tollName": "佛堂收费站",
                                    "entrance": 1,
                                    "controlTypeName": "关闭",
                                }
                            ],
                        },
                    )
                },
            }
        )

        llm_messages = captured_messages["messages"]
        assert isinstance(llm_messages, list)
        assert len(llm_messages) == 3
        assert "路网播报总结助手" in llm_messages[0].content
        assert "| roadCode | highwayName | roadSection | controls | traffic |" in result[
            "final_result"
        ].content
        assert "| G1512 | 甬金高速 | 金华段 | 佛堂收费站，宁波方向入口关闭 | 无 |" in result[
            "final_result"
        ].content

    await engine.dispose()


@pytest.mark.asyncio
async def test_answer_node_skips_generic_context_preparation_for_network_report_with_history(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_create_chat_completion(self: object, **kwargs: object) -> AIMessage:
        del self, kwargs
        return AIMessage(
            content="当前路网整体平稳。",
            response_metadata={"finish_reason": "stop", "model_name": "test-model"},
            usage_metadata={"input_tokens": 21, "output_tokens": 12, "total_tokens": 33},
        )

    async def fail_prepare_context_state(self: object, state: object) -> dict[str, object]:
        del self, state
        raise AssertionError("network_report renderer path should not prepare generic answer context")

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )
    monkeypatch.setattr(
        "app.agent.nodes.answer_node.AnswerNode.prepare_context_state",
        fail_prepare_context_state,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-report-history-short-circuit.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-report-history-short-circuit-001",
                "latest_user_message": "请提供省内整体实时路况总结。",
                "input_messages": [
                    LlmInputMessage(
                        role="user",
                        content="请提供省内整体实时路况总结。",
                    ),
                    LlmInputMessage(
                        role="assistant",
                        content="AI播报总结：上一轮报表内容",
                    ),
                    LlmInputMessage(
                        role="user",
                        content="请提供省内整体实时路况总结。",
                    ),
                ],
                "report_context": "完整 report_content",
                "execution_plan": ExecutionPlan(
                    primary_category="network_report",
                    execution_mode="single_step",
                    recommended_route="report",
                ),
                "step_results": {
                    "report_1": ExecutorResult(
                        step_id="report_1",
                        executor="report",
                        is_success=True,
                        normalized_result={
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [],
                            "exit_top_items": [
                                {
                                    "roadGBCode": "G1512",
                                    "roadName": "G1512甬金高速（金华段）",
                                    "directionName": "宁波方向",
                                    "tollName": "佛堂收费站",
                                    "entrance": 1,
                                    "controlTypeName": "关闭",
                                }
                            ],
                        },
                    )
                },
            }
        )

    assert "| roadCode | highwayName | roadSection | controls | traffic |" in result[
        "final_result"
    ].content
    await engine.dispose()


@pytest.mark.asyncio
async def test_answer_node_limits_network_report_context_to_latest_user_turn(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_prepare_context(self: object, **kwargs: object) -> PreparedContext:
        del self
        captured.update(kwargs)
        return PreparedContext(messages=[], used_session_memory=False)

    monkeypatch.setattr(
        "app.agent.nodes.answer_node.AnswerNode._prepare_context",
        fake_prepare_context,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-report-turns.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        await answer_node.prepare_context_state(
            {
                "session_id": "session-report-turns-001",
                "latest_user_message": "请给我最新路网总结",
                "primary_category": "network_report",
                "step_results": {
                    "report_1": ExecutorResult(
                        step_id="report_1",
                        executor="report",
                        is_success=True,
                        normalized_result={
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [],
                            "exit_top_items": [],
                        },
                    )
                },
            }
        )

    assert captured["max_turns"] == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_answer_node_keeps_network_report_table_even_when_user_only_requests_summary(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_create_chat_completion(self: object, **kwargs: object) -> AIMessage:
        del self, kwargs
        return AIMessage(
            content="当前全网以局部管控为主，甬金高速金华段需重点关注。",
            response_metadata={"finish_reason": "stop", "model_name": "test-model"},
            usage_metadata={"input_tokens": 21, "output_tokens": 12, "total_tokens": 33},
        )

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-report-summary-only.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-report-summary-only-001",
                "latest_user_message": "请提供省内整体实时路况总结。",
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                    report_context="完整 report_content",
                ),
                "execution_plan": ExecutionPlan(
                    primary_category="network_report",
                    execution_mode="single_step",
                    recommended_route="report",
                ),
                "step_results": {
                    "report_1": ExecutorResult(
                        step_id="report_1",
                        executor="report",
                        is_success=True,
                        normalized_result={
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [
                                {
                                    "roadId": "33171",
                                    "roadGBCode": "G1512",
                                    "roadName": "G1512甬金金华段",
                                    "direction": 100701,
                                    "directionName": "宁波方向",
                                    "controlType": 10102,
                                    "controlTypeName": "单向封道",
                                    "des": "因协助地方双江湖湖底隧道施工需要",
                                }
                            ],
                            "exit_top_items": [],
                        },
                    )
                },
            }
        )

        assert result["final_result"].content.startswith(
            "当前全网以局部管控为主，甬金高速金华段需重点关注。"
        )
        assert "| roadCode | highwayName | roadSection | controls | traffic |" in result[
            "final_result"
        ].content
    assert "G1512" in result["final_result"].content

    await engine.dispose()


@pytest.mark.asyncio
async def test_answer_node_renders_network_report_from_serialized_step_results(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_create_chat_completion(self: object, **kwargs: object) -> AIMessage:
        del self, kwargs
        return AIMessage(
            content="The road network is stable.",
            response_metadata={"finish_reason": "stop", "model_name": "test-model"},
            usage_metadata={"input_tokens": 21, "output_tokens": 12, "total_tokens": 33},
        )

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-report-serialized.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-report-serialized-001",
                "latest_user_message": "请提供省内整体实时路况总结。",
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                    report_context="完整 report_content",
                ),
                "execution_plan": ExecutionPlan(
                    primary_category="network_report",
                    execution_mode="single_step",
                    recommended_route="report",
                ),
                "step_results": {
                    "report_1": {
                        "step_id": "report_1",
                        "executor": "report",
                        "is_success": True,
                        "normalized_result": {
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [],
                            "exit_top_items": [
                                {
                                    "roadGBCode": "G1512",
                                    "roadName": "G1512甬金高速（金华段）",
                                    "directionName": "宁波方向",
                                    "tollName": "佛堂收费站",
                                    "entrance": 1,
                                    "controlTypeName": "关闭",
                                }
                            ],
                        },
                    }
                },
            }
        )

    assert "| G1512 |" in result["final_result"].content
    await engine.dispose()


@pytest.mark.asyncio
async def test_answer_node_uses_fixed_summary_instruction_for_network_report(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured_messages: dict[str, object] = {}

    async def fake_create_chat_completion(self: object, **kwargs: object) -> AIMessage:
        del self
        captured_messages.update(kwargs)
        return AIMessage(
            content="当前路网整体平稳。",
            response_metadata={"finish_reason": "stop", "model_name": "test-model"},
            usage_metadata={"input_tokens": 21, "output_tokens": 12, "total_tokens": 33},
        )

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-fixed-summary.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        await answer_node.run(
            {
                "session_id": "session-report-fixed-001",
                "latest_user_message": (
                    "请提供省内整体实时路况总结。\n"
                    "请先输出 1-2 句中文播报总结，再输出 Markdown 表格。\n"
                    "表头固定为：| roadCode | highwayName | roadSection | controls | traffic |\n"
                    "不要输出“序号”等额外列，不要改写表头字段名，不要在表格前重复解释字段含义。\n"
                    "controls 和 traffic 多个值请用、分隔，没有请填无"
                ),
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                    report_context="完整 report_content",
                ),
                "execution_plan": ExecutionPlan(
                    primary_category="network_report",
                    execution_mode="single_step",
                    recommended_route="report",
                ),
                "step_results": {
                    "report_1": {
                        "step_id": "report_1",
                        "executor": "report",
                        "is_success": True,
                        "normalized_result": {
                            "congestion_total_mile": 0,
                            "congestion_top_items": [],
                            "control_top_items": [],
                            "exit_top_items": [],
                        },
                    }
                },
            }
        )

    llm_messages = captured_messages["messages"]
    assert isinstance(llm_messages, list)
    assert len(llm_messages) == 3
    assert llm_messages[2].content == "请基于完整报表上下文输出1-2句路网播报总结。"
    await engine.dispose()
