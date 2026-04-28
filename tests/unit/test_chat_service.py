"""ChatService 单元测试。"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessageChunk

from app.agent.state import ChatTurnResult
from app.core.exceptions import AppException
from app.schemas.openai_compat import OpenAIChatCompletionRequest


def _install_mcp_test_stubs() -> None:
    """为单测安装最小化 mcp 依赖桩。"""

    if "mcp" in sys.modules:
        return

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def initialize(self) -> None:
            return None

    class _DummyImplementation:
        def __init__(self, name: str, version: str) -> None:
            self.name = name
            self.version = version

    class _DummyTextContent:
        text = ""

        def model_dump_json(self) -> str:
            return "{}"

    class _DummyCallToolResult:
        content: list[object] = []
        structuredContent = None
        isError = False

    async def _unused_async_context(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("mcp stub should not be executed in chat_service unit tests")
        yield None

    mcp_module = types.ModuleType("mcp")
    mcp_module.ClientSession = _DummyClientSession
    mcp_module.types = types.SimpleNamespace(
        Implementation=_DummyImplementation,
        CallToolResult=_DummyCallToolResult,
        TextContent=_DummyTextContent,
    )

    mcp_client_module = types.ModuleType("mcp.client")
    mcp_streamable_http_module = types.ModuleType("mcp.client.streamable_http")
    mcp_streamable_http_module.streamablehttp_client = _unused_async_context
    mcp_sse_module = types.ModuleType("mcp.client.sse")
    mcp_sse_module.sse_client = _unused_async_context
    mcp_stdio_module = types.ModuleType("mcp.client.stdio")
    mcp_stdio_module.stdio_client = _unused_async_context
    mcp_stdio_module.StdioServerParameters = object

    sys.modules["mcp"] = mcp_module
    sys.modules["mcp.client"] = mcp_client_module
    sys.modules["mcp.client.streamable_http"] = mcp_streamable_http_module
    sys.modules["mcp.client.sse"] = mcp_sse_module
    sys.modules["mcp.client.stdio"] = mcp_stdio_module


_install_mcp_test_stubs()


def _build_service(db_session: AsyncMock):
    """在安装 mcp stub 后构造 ChatService。"""

    from app.services.chat_service import ChatService

    return ChatService(db_session)


def _build_chat_request() -> OpenAIChatCompletionRequest:
    """构造稳定的聊天请求。"""

    return OpenAIChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "你好"}],
    )


def _build_turn_result(
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
) -> ChatTurnResult:
    """构造稳定的最终结果。"""

    return ChatTurnResult(
        session_id="session-001",
        content="测试模型回答：你好",
        model_name="test-model",
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
        finish_reason=finish_reason,
        route="answer",
        reasoning_content=reasoning_content,
    )


def test_build_execution_request_uses_nested_chat_template_kwargs_enable_thinking() -> None:
    """Nested chat_template_kwargs.enable_thinking should reach the execution request."""

    db_session = AsyncMock()
    service = _build_service(db_session)
    request = OpenAIChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "浣犲ソ"}],
        chat_template_kwargs={"enable_thinking": False},
    )

    execution_request = service._build_execution_request(
        chat_request=request,
        session_id=None,
    )

    assert execution_request.enable_thinking is False


def test_build_execution_request_prefers_top_level_enable_thinking() -> None:
    """Top-level enable_thinking should win over nested chat_template_kwargs."""

    db_session = AsyncMock()
    service = _build_service(db_session)
    request = OpenAIChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "浣犲ソ"}],
        enable_thinking=True,
        chat_template_kwargs={"enable_thinking": False},
    )

    execution_request = service._build_execution_request(
        chat_request=request,
        session_id=None,
    )

    assert execution_request.enable_thinking is True


def test_build_execution_request_forwards_scheduled_route() -> None:
    """scheduled_route should be copied into the internal execution request."""

    db_session = AsyncMock()
    service = _build_service(db_session)
    request = OpenAIChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "请提供省内整体实时路况总结。"}],
        scheduled_route="report",
    )

    execution_request = service._build_execution_request(
        chat_request=request,
        session_id=None,
    )

    assert execution_request.scheduled_route == "report"


def test_build_execution_request_forwards_brief_answer() -> None:
    """brief_answer should reach answer_node through the internal execution request."""

    db_session = AsyncMock()
    service = _build_service(db_session)
    request = OpenAIChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "杭金衢高速堵不堵"}],
        brief_answer=True,
    )

    execution_request = service._build_execution_request(
        chat_request=request,
        session_id=None,
    )

    assert execution_request.brief_answer is True


def test_build_execution_request_defaults_brief_answer_to_true() -> None:
    """Omitted brief_answer should use compact answer mode by default."""

    db_session = AsyncMock()
    service = _build_service(db_session)
    request = OpenAIChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "traffic status"}],
    )

    execution_request = service._build_execution_request(
        chat_request=request,
        session_id=None,
    )

    assert execution_request.brief_answer is True


def test_build_execution_request_forwards_brief_answer_false() -> None:
    """Explicit brief_answer=false should disable compact answer mode."""

    db_session = AsyncMock()
    service = _build_service(db_session)
    request = OpenAIChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "traffic status"}],
        brief_answer=False,
    )

    execution_request = service._build_execution_request(
        chat_request=request,
        session_id=None,
    )

    assert execution_request.brief_answer is False


@pytest.mark.asyncio
async def test_prepare_chat_execution_resolves_session_and_persists_user_message_once() -> None:
    """公共准备流程应解析会话并且只持久化一次用户消息。"""

    db_session = AsyncMock()
    service = _build_service(db_session)
    service._ensure_session = AsyncMock(return_value="session-001")  # type: ignore[method-assign]
    service._persist_user_message = AsyncMock()  # type: ignore[method-assign]

    prepared = await service._prepare_chat_execution(
        chat_request=_build_chat_request(),
        session_id=None,
        commit_after_prepare=False,
    )

    service._ensure_session.assert_awaited_once()
    service._persist_user_message.assert_awaited_once()
    db_session.commit.assert_not_awaited()
    assert prepared.resolved_session_id == "session-001"
    assert prepared.execution_request.session_id == "session-001"
    assert prepared.prepare_duration_ms >= 0


@pytest.mark.asyncio
async def test_consume_graph_events_hides_tool_node_reasoning_text() -> None:
    """tool_node 轮次应隐藏推理文本，但保留 tool_calls 与结束信号。"""

    db_session = AsyncMock()
    service = _build_service(db_session)

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                yield {"event": "on_chain_start", "name": "tool_node", "data": {}}
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content="内部推理文本",
                            tool_call_chunks=[
                                {
                                    "index": 0,
                                    "id": "call_calculator",
                                    "name": "calculator",
                                    "args": '{"expression":"1+1"}',
                                }
                            ],
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content="",
                            response_metadata={
                                "finish_reason": "tool_calls",
                                "model_name": "test-model",
                            },
                        )
                    },
                }
                yield {"event": "on_chain_end", "name": "tool_node", "data": {}}

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    payloads = [
        payload
        async for payload in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-001",
            request_start_time=0.0,
            prepare_duration_ms=1.0,
        )
    ]
    response_body = "".join(payloads)

    assert "内部推理文本" not in response_body
    assert '"tool_calls"' in response_body
    assert '"finish_reason": "tool_calls"' in response_body
    assert "[DONE]" in response_body


@pytest.mark.asyncio
async def test_consume_graph_events_saves_checkpoint_after_stream_success() -> None:
    """流式成功结束后应提交事务并保存 checkpoint。"""

    db_session = AsyncMock()
    service = _build_service(db_session)
    service._session_repository = AsyncMock()
    service._save_checkpoint_safely = AsyncMock()  # type: ignore[method-assign]

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                yield {"event": "on_chain_start", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content="测试模型回答：你好",
                            response_metadata={"model_name": "test-model"},
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content="",
                            response_metadata={
                                "finish_reason": "stop",
                                "model_name": "test-model",
                            },
                        )
                    },
                }
                yield {"event": "on_chain_end", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "data": {
                        "output": {
                            "final_result": _build_turn_result(),
                            "checkpoint_payload": {"checkpoint_id": "cp-001"},
                        }
                    },
                }

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    payloads = [
        payload
        async for payload in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-001",
            request_start_time=0.0,
            prepare_duration_ms=2.0,
        )
    ]

    assert "[DONE]" in "".join(payloads)
    service._session_repository.update_timestamp.assert_awaited_once_with("session-001")
    db_session.commit.assert_awaited_once()
    service._save_checkpoint_safely.assert_awaited_once_with({"checkpoint_id": "cp-001"})


@pytest.mark.asyncio
async def test_consume_graph_events_emits_final_result_when_no_stream_chunks() -> None:
    """如果图执行成功但没有任何增量 chunk，仍应把 final_result 内容通过 SSE 发出。"""

    db_session = AsyncMock()
    service = _build_service(db_session)
    service._session_repository = AsyncMock()
    service._save_checkpoint_safely = AsyncMock()  # type: ignore[method-assign]

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                yield {"event": "on_chain_start", "name": "answer_node", "data": {}}
                yield {"event": "on_chain_end", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "data": {
                        "output": {
                            "final_result": _build_turn_result(),
                            "checkpoint_payload": {"checkpoint_id": "cp-final-only"},
                        }
                    },
                }

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    payloads = [
        payload
        async for payload in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-final-only",
            request_start_time=0.0,
            prepare_duration_ms=1.0,
        )
    ]
    response_body = "".join(payloads)

    assert '"content": "测试模型回答：你好"' in response_body
    assert '"finish_reason": "stop"' in response_body
    assert response_body.endswith("data: [DONE]\n\n")
    service._session_repository.update_timestamp.assert_awaited_once_with("session-001")
    db_session.commit.assert_awaited_once()
    service._save_checkpoint_safely.assert_awaited_once_with({"checkpoint_id": "cp-final-only"})


@pytest.mark.asyncio
async def test_consume_graph_events_emits_table_suffix_after_streamed_summary() -> None:
    """前面已经流出总结时，final_result 中额外的表格尾巴仍应在 DONE 前补发。"""

    db_session = AsyncMock()
    service = _build_service(db_session)
    service._session_repository = AsyncMock()
    service._save_checkpoint_safely = AsyncMock()  # type: ignore[method-assign]

    summary_text = "当前全路网监测到12处收费站管控、11处主线管制、3处缓行事件，整体以局部异常为主。"
    table_markdown = (
        "\n\n| roadCode | highwayName | roadSection | controls | traffic |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| G1512 | 甬金高速 | 金华段 | 佛堂收费站，宁波方向入口关闭、出口分流 | 无 |"
    )
    final_text = f"{summary_text}{table_markdown}"

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                yield {"event": "on_chain_start", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content=summary_text,
                            response_metadata={"model_name": "test-model"},
                        )
                    },
                }
                yield {"event": "on_chain_end", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "data": {
                        "output": {
                            "final_result": ChatTurnResult(
                                session_id="session-001",
                                content=final_text,
                                model_name="test-model",
                                prompt_tokens=12,
                                completion_tokens=8,
                                total_tokens=20,
                                finish_reason="stop",
                                route="answer",
                            ),
                            "checkpoint_payload": {"checkpoint_id": "cp-report-tail"},
                        }
                    },
                }

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    payloads = [
        payload
        async for payload in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-report-tail",
            request_start_time=0.0,
            prepare_duration_ms=1.0,
        )
    ]
    response_body = "".join(payloads)

    assert response_body.count(summary_text) == 1
    assert "| roadCode | highwayName | roadSection | controls | traffic |" in response_body
    assert response_body.index("| roadCode | highwayName | roadSection | controls | traffic |") > response_body.index(summary_text)
    assert response_body.rfind("| roadCode | highwayName | roadSection | controls | traffic |") < response_body.rfind('"finish_reason": "stop"')
    assert response_body.endswith("data: [DONE]\n\n")
    service._session_repository.update_timestamp.assert_awaited_once_with("session-001")
    db_session.commit.assert_awaited_once()
    service._save_checkpoint_safely.assert_awaited_once_with({"checkpoint_id": "cp-report-tail"})


@pytest.mark.asyncio
async def test_consume_graph_events_emits_table_suffix_when_streamed_summary_is_unprefixed() -> None:
    """如果流里只发了原始摘要，而 final_result 加了播报前缀，也应补发表格尾巴。"""

    db_session = AsyncMock()
    service = _build_service(db_session)
    service._session_repository = AsyncMock()
    service._save_checkpoint_safely = AsyncMock()  # type: ignore[method-assign]

    streamed_summary = "当前全路网监测到12处收费站管控、11处主线管制、3处缓行事件，整体以局部异常为主。"
    final_text = (
        f"{streamed_summary}\n\n"
        "| roadCode | highwayName | roadSection | controls | traffic |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| G1512 | 甬金高速 | 金华段 | 佛堂收费站，宁波方向入口关闭、出口分流 | 无 |"
    )

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                yield {"event": "on_chain_start", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content=streamed_summary,
                            response_metadata={"model_name": "test-model"},
                        )
                    },
                }
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "data": {
                        "output": {
                            "final_result": ChatTurnResult(
                                session_id="session-001",
                                content=final_text,
                                model_name="test-model",
                                prompt_tokens=12,
                                completion_tokens=8,
                                total_tokens=20,
                                finish_reason="stop",
                                route="answer",
                            ),
                            "checkpoint_payload": {"checkpoint_id": "cp-report-tail-prefix"},
                        }
                    },
                }

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    payloads = [
        payload
        async for payload in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-report-tail-prefix",
            request_start_time=0.0,
            prepare_duration_ms=1.0,
        )
    ]
    response_body = "".join(payloads)

    assert response_body.count(streamed_summary) == 1
    assert "| roadCode | highwayName | roadSection | controls | traffic |" in response_body
    assert response_body.endswith("data: [DONE]\n\n")
    service._session_repository.update_timestamp.assert_awaited_once_with("session-001")
    db_session.commit.assert_awaited_once()
    service._save_checkpoint_safely.assert_awaited_once_with({"checkpoint_id": "cp-report-tail-prefix"})


@pytest.mark.asyncio
async def test_consume_graph_events_accepts_serialized_final_result() -> None:
    """如果 root output 里的 final_result 被序列化成 dict，流式补表也应继续工作。"""

    db_session = AsyncMock()
    service = _build_service(db_session)
    service._session_repository = AsyncMock()
    service._save_checkpoint_safely = AsyncMock()  # type: ignore[method-assign]

    summary_text = "当前全路网监测到1处收费站管控，整体以局部异常为主。"
    final_text = f"{summary_text}\n\n| roadCode | highwayName | roadSection | controls | traffic |\n| --- | --- | --- | --- | --- |\n| G1512 | 甬金高速 | 金华段 | 佛堂收费站，宁波方向入口关闭、出口分流 | 无 |"

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                yield {"event": "on_chain_start", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content=summary_text,
                            response_metadata={"model_name": "test-model"},
                        )
                    },
                }
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "data": {
                        "output": {
                            "final_result": {
                                "session_id": "session-001",
                                "content": final_text,
                                "model_name": "test-model",
                                "prompt_tokens": 12,
                                "completion_tokens": 8,
                                "total_tokens": 20,
                                "finish_reason": "stop",
                                "route": "answer",
                            },
                            "checkpoint_payload": {"checkpoint_id": "cp-serialized-final"},
                        }
                    },
                }

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    payloads = [
        payload
        async for payload in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-serialized-final",
            request_start_time=0.0,
            prepare_duration_ms=1.0,
        )
    ]
    response_body = "".join(payloads)

    assert summary_text in response_body
    assert "| roadCode | highwayName | roadSection | controls | traffic |" in response_body
    assert response_body.endswith("data: [DONE]\n\n")
    service._session_repository.update_timestamp.assert_awaited_once_with("session-001")
    db_session.commit.assert_awaited_once()
    service._save_checkpoint_safely.assert_awaited_once_with({"checkpoint_id": "cp-serialized-final"})


@pytest.mark.asyncio
async def test_consume_graph_events_streams_error_after_first_payload() -> None:
    """首个 payload 之后异常时应输出流式错误并补 DONE。"""

    db_session = AsyncMock()
    service = _build_service(db_session)

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                yield {"event": "on_chain_start", "name": "answer_node", "data": {}}
                yield {
                    "event": "on_chat_model_stream",
                    "name": "FakeLLM",
                    "data": {
                        "chunk": AIMessageChunk(
                            content="测试",
                            response_metadata={"model_name": "test-model"},
                        )
                    },
                }
                raise AppException("流式失败", error_code="stream_failure")
                yield {"event": "on_chain_end", "name": "answer_node", "data": {}}

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    payloads = [
        payload
        async for payload in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-001",
            request_start_time=0.0,
            prepare_duration_ms=1.0,
        )
    ]
    response_body = "".join(payloads)

    assert '"content": "测试"' in response_body
    assert '"error"' in response_body
    assert '"stream_failure"' in response_body
    assert response_body.endswith("data: [DONE]\n\n")
    db_session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_graph_events_raises_before_first_payload() -> None:
    """首个 payload 之前异常时应直接抛出异常。"""

    db_session = AsyncMock()
    service = _build_service(db_session)

    class _FakeGraph:
        def stream_events(self, execution_request: object) -> AsyncIterator[dict[str, object]]:
            del execution_request

            async def _iterator() -> AsyncIterator[dict[str, object]]:
                raise AppException("首块前失败", error_code="before_first_payload")
                yield {"event": "on_chain_start", "name": "answer_node", "data": {}}

            return _iterator()

    service._conversation_graph = _FakeGraph()  # type: ignore[assignment]

    with pytest.raises(AppException, match="首块前失败"):
        async for _ in service._consume_graph_events(
            execution_request=SimpleNamespace(session_id="session-001", model_name="test-model"),
            request_id="req-001",
            request_start_time=0.0,
            prepare_duration_ms=1.0,
        ):
            pass

    db_session.rollback.assert_awaited_once()
