"""测试配置文件。
负责补齐测试运行时的项目根目录导入路径，并提供基础测试夹具。
当前阶段不负责复杂测试环境编排和外部依赖容器管理。
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.clients.llm_client import (
    LlmChatCompletionChunk,
    LlmChatCompletionResult,
    LlmToolCall,
    LlmToolCallChunk,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """在每个测试前后清理配置缓存，避免环境变量互相污染。"""

    from app.core.config import get_settings
    from app.persistence.database import clear_database_caches

    get_settings.cache_clear()
    clear_database_caches()
    yield
    get_settings.cache_clear()
    clear_database_caches()


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[TestClient]:
    """提供使用临时 SQLite 数据库和假 LLM 的 FastAPI 测试客户端。"""

    sqlite_database_path = tmp_path / "integration-test.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("POSTGRES_DSN", f"sqlite+aiosqlite:///{sqlite_database_path.as_posix()}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    async def fake_create_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> LlmChatCompletionResult:
        """为集成测试返回稳定的假模型回答与工具调用。"""

        del self, tool_choice
        latest_user_message = ""
        latest_tool_output = ""

        for message in reversed(messages):
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
            if role == "tool" and not latest_tool_output:
                latest_tool_output = content
            if role == "user" and not latest_user_message:
                latest_user_message = content

        need_calculator_tool = "1+1" in latest_user_message or "计算" in latest_user_message
        if tools and not latest_tool_output and need_calculator_tool:
            return LlmChatCompletionResult(
                content="",
                model_name=model_name or "test-model",
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                finish_reason="tool_calls",
                tool_calls=[
                    LlmToolCall(
                        tool_call_id="call_calculator",
                        tool_name="calculator",
                        arguments={"expression": "1+1"},
                    )
                ],
            )

        need_datetime_tool = "时间" in latest_user_message or "几点" in latest_user_message
        if tools and not latest_tool_output and need_datetime_tool:
            return LlmChatCompletionResult(
                content="",
                model_name=model_name or "test-model",
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                finish_reason="tool_calls",
                tool_calls=[
                    LlmToolCall(
                        tool_call_id="call_datetime",
                        tool_name="current_datetime",
                        arguments={"timezone_name": "Asia/Shanghai"},
                    )
                ],
            )

        if latest_tool_output:
            return LlmChatCompletionResult(
                content=f"测试模型回答：工具结果是 {latest_tool_output}",
                model_name=model_name or "test-model",
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                finish_reason="stop",
            )

        return LlmChatCompletionResult(
            content=f"测试模型回答：{latest_user_message}",
            model_name=model_name or "test-model",
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            finish_reason="stop",
        )

    def fake_stream_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> AsyncIterator[LlmChatCompletionChunk]:
        """为集成测试返回真实逐块产生的假流式结果。"""

        del self, tool_choice
        latest_user_message = ""

        for message in reversed(messages):
            if getattr(message, "role", "") == "user":
                latest_user_message = getattr(message, "content", "")
                break

        async def iterator() -> AsyncIterator[LlmChatCompletionChunk]:
            resolved_model_name = model_name or "test-model"
            if tools and ("1+1" in latest_user_message or "计算" in latest_user_message):
                yield LlmChatCompletionChunk(
                    model_name=resolved_model_name,
                    tool_call_chunks=[
                        LlmToolCallChunk(
                            index=0,
                            tool_call_id="call_calculator",
                            tool_name="calculator",
                            arguments_chunk='{"expression":"1+1"}',
                        )
                    ],
                )
                yield LlmChatCompletionChunk(
                    model_name=resolved_model_name,
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                    finish_reason="tool_calls",
                )
                return

            full_text = f"测试模型回答：{latest_user_message}"
            split_index = max(1, len(full_text) // 2)
            yield LlmChatCompletionChunk(
                content_delta=full_text[:split_index],
                model_name=resolved_model_name,
            )
            yield LlmChatCompletionChunk(
                content_delta=full_text[split_index:],
                model_name=resolved_model_name,
            )
            yield LlmChatCompletionChunk(
                model_name=resolved_model_name,
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                finish_reason="stop",
            )

        return iterator()

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.stream_chat_completion",
        fake_stream_chat_completion,
    )

    from app.main import create_app

    application = create_app()

    with TestClient(application) as client:
        yield client
