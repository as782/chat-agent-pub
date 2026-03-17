"""测试配置文件。
负责补齐测试运行时的项目根目录导入路径，并提供基础测试夹具。
当前阶段不负责复杂测试环境编排和外部依赖容器管理。
"""

from __future__ import annotations

import json
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


@pytest.fixture(autouse=True)
def isolate_mcp_servers_env(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """为测试提供稳定的 MCP 默认配置，避免受本地 .env 污染。"""

    monkeypatch.setenv("MCP_SERVERS_JSON", "[]")
    yield


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[TestClient]:
    """提供使用临时 SQLite 数据库和假 LLM 的 FastAPI 测试客户端。"""

    sqlite_database_path = tmp_path / "integration-test.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("POSTGRES_DSN", f"sqlite+aiosqlite:///{sqlite_database_path.as_posix()}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("RAGFLOW_API_KEY", "test-ragflow-key")
    monkeypatch.setenv("RAGFLOW_BASE_URL", "https://ragflow.example.com")
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            [
                {
                    "name": "demo-mcp-http",
                    "transport": "http",
                    "endpoint": "https://mcp.example.com",
                }
            ],
            ensure_ascii=False,
        ),
    )

    async def fake_create_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> LlmChatCompletionResult:
        """为集成测试返回稳定的假模型回答与工具调用。"""

        del self, tool_choice, enable_thinking
        latest_user_message = ""
        latest_tool_output = ""
        all_message_contents: list[str] = []
        user_messages: list[str] = []
        available_tool_names: list[str] = []

        for tool in tools or []:
            if isinstance(tool, dict):
                function_payload = tool.get("function", {})
                if isinstance(function_payload, dict) and isinstance(
                    function_payload.get("name"),
                    str,
                ):
                    available_tool_names.append(str(function_payload["name"]))
            else:
                available_tool_names.append(str(getattr(tool, "name", "")))

        for message in reversed(messages):
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
            all_message_contents.append(str(content))
            if role == "tool" and not latest_tool_output:
                latest_tool_output = content
            if role == "user" and not latest_user_message:
                latest_user_message = content
            if role == "user":
                user_messages.append(str(content))

        history_contains_name = any("我叫小王" in message for message in user_messages[1:])
        explicit_force_no_memory = any(
            "如果不知道就说不知道" in message for message in all_message_contents
        )
        has_knowledge_context = any(
            "以下是知识库检索结果" in message for message in all_message_contents
        )
        has_mcp_context = any(
            (
                "以下是当前系统已配置的 MCP 服务骨架信息" in message
                or "以下是当前系统已接入的 MCP 服务与工具信息" in message
            )
            for message in all_message_contents
        )

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

        mcp_weather_tool_name = next(
            (
                tool_name
                for tool_name in available_tool_names
                if tool_name.startswith("mcp_") and "weather" in tool_name
            ),
            None,
        )
        if mcp_weather_tool_name and not latest_tool_output and "天气" in latest_user_message:
            return LlmChatCompletionResult(
                content="",
                model_name=model_name or "test-model",
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                finish_reason="tool_calls",
                tool_calls=[
                    LlmToolCall(
                        tool_call_id="call_mcp_weather",
                        tool_name=mcp_weather_tool_name,
                        arguments={"city": "杭州"},
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

        if "我刚刚告诉你的名字是什么" in latest_user_message:
            if history_contains_name:
                return LlmChatCompletionResult(
                    content="测试模型回答：你刚刚说你叫小王",
                    model_name=model_name or "test-model",
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                    finish_reason="stop",
                )
            if explicit_force_no_memory:
                return LlmChatCompletionResult(
                    content="测试模型回答：不知道",
                    model_name=model_name or "test-model",
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                    finish_reason="stop",
                )

        if has_knowledge_context and "西湖" in latest_user_message:
            return LlmChatCompletionResult(
                content="测试模型回答：根据知识库，西湖位于杭州。",
                model_name=model_name or "test-model",
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
                finish_reason="stop",
            )
        if has_mcp_context and "MCP" in latest_user_message.upper():
            return LlmChatCompletionResult(
                content="测试模型回答：当前已配置 MCP 服务骨架。",
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
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[LlmChatCompletionChunk]:
        """为集成测试返回真实逐块产生的假流式结果。"""

        del self, tool_choice, enable_thinking
        latest_user_message = ""
        user_messages: list[str] = []
        all_message_contents: list[str] = []
        available_tool_names: list[str] = []

        for tool in tools or []:
            if isinstance(tool, dict):
                function_payload = tool.get("function", {})
                if isinstance(function_payload, dict) and isinstance(
                    function_payload.get("name"),
                    str,
                ):
                    available_tool_names.append(str(function_payload["name"]))
            else:
                available_tool_names.append(str(getattr(tool, "name", "")))

        for message in reversed(messages):
            if getattr(message, "role", "") == "user":
                latest_user_message = getattr(message, "content", "")
                break
        for message in messages:
            role = getattr(message, "role", "")
            content = str(getattr(message, "content", ""))
            all_message_contents.append(content)
            if role == "user":
                user_messages.append(content)

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

            mcp_weather_tool_name = next(
                (
                    tool_name
                    for tool_name in available_tool_names
                    if tool_name.startswith("mcp_") and "weather" in tool_name
                ),
                None,
            )
            if mcp_weather_tool_name and "天气" in latest_user_message:
                yield LlmChatCompletionChunk(
                    model_name=resolved_model_name,
                    tool_call_chunks=[
                        LlmToolCallChunk(
                            index=0,
                            tool_call_id="call_mcp_weather",
                            tool_name=mcp_weather_tool_name,
                            arguments_chunk='{"city":"杭州"}',
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

            if any("以下是知识库检索结果" in message for message in all_message_contents) and (
                "西湖" in latest_user_message
            ):
                full_text = "测试模型回答：根据知识库，西湖位于杭州。"
            elif any(
                (
                    "以下是当前系统已配置的 MCP 服务骨架信息" in message
                    or "以下是当前系统已接入的 MCP 服务与工具信息" in message
                )
                for message in all_message_contents
            ) and ("MCP" in latest_user_message.upper()):
                full_text = "测试模型回答：当前已配置 MCP 服务骨架。"
            elif "我刚刚告诉你的名字是什么" in latest_user_message:
                if any("我叫小王" in message for message in user_messages[:-1]):
                    full_text = "测试模型回答：你刚刚说你叫小王"
                elif any("如果不知道就说不知道" in message for message in all_message_contents):
                    full_text = "测试模型回答：不知道"
                else:
                    full_text = f"测试模型回答：{latest_user_message}"
            else:
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
