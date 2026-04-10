"""回答节点单元测试。"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.nodes.answer_node import AnswerNode
from app.agent.state import ExecutionPlan, ExecutionStep, ExecutorResult, PreparedContext
from app.clients.llm_client import LlmInputMessage
from app.persistence.base import Base
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall


def test_answer_node_builds_executor_results_context() -> None:
    """统一 step_results 应被整理成可注入模型的上下文文本。"""

    context = AnswerNode._build_executor_results_context(
        {
            "rag_1": ExecutorResult(
                step_id="rag_1",
                executor="rag",
                is_success=True,
                normalized_result={"result_count": 2, "sources": ["doc-1", "doc-2"]},
                summary="知识检索命中 2 条结果。",
            ),
            "report_1": ExecutorResult(
                step_id="report_1",
                executor="report",
                is_success=True,
                normalized_result={"query_time": "2026-03-31 09:00:00", "congestion_total_mile": 12.5},
                summary="已整理路网报告任务参数。",
            ),
        }
    )

    assert context is not None
    assert "[rag_1] executor=rag success=True" in context
    assert "知识检索命中 2 条结果。" in context
    assert "12.5" in context


def test_answer_node_resolves_prompt_name_from_category() -> None:
    assert (
        AnswerNode._resolve_answer_prompt_name({"primary_category": "traffic_status"})
        == "TRAFFIC_SUMMARY_PROMPT"
    )
    assert (
        AnswerNode._resolve_answer_prompt_name({"primary_category": "network_report"})
        == "NETWORK_REPORT_SUMMARY_PROMPT"
    )


def test_answer_node_uses_composite_prompt_for_route_congestion_questions() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "杭州到金华堵不堵",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
            "traffic_1": ExecutorResult(
                step_id="traffic_1",
                executor="traffic",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "COMPOSITE_ANSWER_PROMPT"


def test_answer_node_keeps_traffic_prompt_for_traffic_only_questions() -> None:
    state = {
        "primary_category": "traffic_status",
        "latest_user_message": "杭金衢高速堵不堵",
        "step_results": {
            "traffic_1": ExecutorResult(
                step_id="traffic_1",
                executor="traffic",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "TRAFFIC_SUMMARY_PROMPT"


def test_answer_node_keeps_route_prompt_for_route_only_questions() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "杭州到金华怎么走",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "ROUTE_SUMMARY_PROMPT"


def test_answer_node_uses_composite_prompt_for_route_and_rag_results() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "今天上高速到明天下高速要过路费吗",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
            "rag_1": ExecutorResult(
                step_id="rag_1",
                executor="rag",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "COMPOSITE_ANSWER_PROMPT"


def test_answer_node_builds_toll_focused_composite_instruction() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "今天上高速到明天下高速要过路费吗",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
            "rag_1": ExecutorResult(
                step_id="rag_1",
                executor="rag",
                is_success=True,
            ),
        },
    }

    instruction = AnswerNode._resolve_answer_instruction(state)

    assert "收费判断" in instruction
    assert "rag、route" in instruction


@pytest.mark.asyncio
async def test_answer_node_reuses_tool_completion_result_without_new_llm_call(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """工具节点已得到完成结果时，回答节点应直接收口而不是再次调用模型。"""

    async def fail_create_chat_completion(*args, **kwargs) -> None:
        """如果进入普通回答分支，测试应直接失败。"""

        del args, kwargs
        raise AssertionError("answer_node 不应在已有 tool_completion_result 时再次调用 LLM")

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fail_create_chat_completion,
    )

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'answer-node.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-001",
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                ),
                "tool_completion_result": AIMessage(
                    content="测试模型回答：工具结果是 2",
                    response_metadata={"finish_reason": "stop"},
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                ),
                "executed_tool_calls": [
                    ExecutedToolCall(
                        tool_call_id="call_calculator",
                        tool_name="calculator",
                        arguments={"expression": "1+1"},
                        output="2",
                    )
                ],
            }
        )

        persisted_messages = await MessageRepository(db_session).list_by_session("session-001")

        assert result["final_result"].content == "测试模型回答：工具结果是 2"
        assert result["final_result"].tool_calls[0].tool_name == "calculator"
        assert [message.role for message in persisted_messages] == ["assistant"]
        assert persisted_messages[0].content == "测试模型回答：工具结果是 2"

    await engine.dispose()

async def test_answer_node_regenerates_summary_for_multi_step_answer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """多步骤计划进入 answer 步时，应基于累计结果重新总结，而不是复用上一轮工具补全。"""

    async def fake_create_chat_completion(*args, **kwargs) -> AIMessage:
        """返回稳定的总结结果。"""

        del args, kwargs
        return AIMessage(
            content="测试模型回答：根据政策和路线结果生成统一总结。",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 16, "output_tokens": 12, "total_tokens": 28},
        )

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )

    def fake_create_runnable(self: object, **kwargs: object) -> object:
        """让回答节点通过 astream 获取稳定总结结果。"""

        del self, kwargs

        class _FakeRunnable:
            async def astream(self, messages: list[object], config=None):
                del messages, config
                ai_message = await fake_create_chat_completion()
                yield AIMessageChunk(
                    content=ai_message.content,
                    response_metadata=ai_message.response_metadata or {},
                    usage_metadata=ai_message.usage_metadata or {},
                    tool_call_chunks=[],
                )

        return _FakeRunnable()

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fake_create_runnable,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-summary.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-002",
                "current_step_id": "answer_1",
                "execution_plan": ExecutionPlan(
                    primary_category="route_planning",
                    execution_mode="multi_step",
                    recommended_route="route",
                    steps=[
                        ExecutionStep(
                            step_id="rag_1",
                            executor="rag",
                            goal="检索政策标准",
                        ),
                        ExecutionStep(
                            step_id="route_1",
                            executor="route",
                            goal="查询路线方案",
                        ),
                        ExecutionStep(
                            step_id="answer_1",
                            executor="answer",
                            goal="汇总前置结果",
                            depends_on=["rag_1", "route_1"],
                        ),
                    ],
                ),
                "step_results": {
                    "rag_1": ExecutorResult(
                        step_id="rag_1",
                        executor="rag",
                        is_success=True,
                        normalized_result={"sources": ["policy-doc"]},
                    ),
                    "route_1": ExecutorResult(
                        step_id="route_1",
                        executor="route",
                        is_success=True,
                        normalized_result={"origin": "杭州", "destination": "金华"},
                    ),
                },
                "prepared_context": PreparedContext(
                    messages=[LlmInputMessage(role="system", content="请总结前置结果。")],
                    used_session_memory=False,
                ),
                "tool_completion_result": AIMessage(
                    content="测试模型回答：工具阶段临时结果。",
                    response_metadata={"finish_reason": "stop"},
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                ),
                "executed_tool_calls": [
                    ExecutedToolCall(
                        tool_call_id="call_route_plan",
                        tool_name="mcp_demo_http__route_plan",
                        arguments={"origin": "杭州", "destination": "金华"},
                        output="杭州到金华推荐走高速。",
                    )
                ],
            }
        )

        persisted_messages = await MessageRepository(db_session).list_by_session("session-002")

        assert result["final_result"].content == "测试模型回答：根据政策和路线结果生成统一总结。"
        assert result["final_result"].tool_calls[0].tool_name == "mcp_demo_http__route_plan"
        assert [message.role for message in persisted_messages] == ["assistant"]
        assert persisted_messages[0].content == "测试模型回答：根据政策和路线结果生成统一总结。"

    await engine.dispose()
