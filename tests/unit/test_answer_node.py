"""回答节点单元测试。"""

from pathlib import Path

import pytest
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.nodes.answer_node import AnswerNode
from app.agent.state import ExecutorResult, PreparedContext
from app.clients.llm_client import LlmChatCompletionResult
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
                normalized_result={"scope": "全路网", "need_table": True},
                summary="已整理路网报告任务参数。",
            ),
        }
    )

    assert context is not None
    assert "[rag_1] executor=rag success=True" in context
    assert "知识检索命中 2 条结果。" in context
    assert "全路网" in context


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
                "tool_completion_result": LlmChatCompletionResult(
                    content="测试模型回答：工具结果是 2",
                    model_name="test-model",
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                    finish_reason="stop",
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
