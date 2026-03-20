"""测试配置文件。

负责补齐测试运行时的项目导入路径，并为集成测试提供稳定的 SQLite、假 LLM、
假 MCP 环境。当前阶段不负责复杂容器编排和真实第三方联调。
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pytest import MonkeyPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """在每个测试前后清理配置缓存，避免环境变量相互污染。"""

    from app.core.config import get_settings
    from app.persistence.database import clear_database_caches

    get_settings.cache_clear()
    clear_database_caches()
    yield
    get_settings.cache_clear()
    clear_database_caches()


@pytest.fixture(autouse=True)
def isolate_mcp_servers_env(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """为测试提供稳定的 MCP 默认配置，避免本地 .env 干扰。"""

    monkeypatch.setenv("MCP_SERVERS_JSON", "[]")
    yield


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[TestClient]:
    """提供使用临时 SQLite 与假 LLM 的 FastAPI 测试客户端。"""

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

    def _resolve_msg_type(message: object) -> str:
        """统一获取消息类型，兼容 LlmInputMessage(.role) 和 BaseMessage(.type)。"""
        msg_type = getattr(message, "type", None)
        if msg_type:
            return str(msg_type)
        role = getattr(message, "role", "")
        if role == "user":
            return "human"
        if role == "assistant":
            return "ai"
        return str(role)

    async def fake_create_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AIMessage:
        """为集成测试返回稳定的假模型结果。"""

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
            msg_type = _resolve_msg_type(message)
            content = str(getattr(message, "content", ""))
            all_message_contents.append(content)

            if msg_type in ("tool", "function") and not latest_tool_output:
                latest_tool_output = content
            if msg_type == "human" and not latest_user_message:
                latest_user_message = content
            if msg_type == "human":
                user_messages.append(content)

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

        is_planner = any("生成分类与执行计划" in message for message in all_message_contents)
        if is_planner:
            plan_json = '{"primary_category": "general", "steps": [{"executor": "answer"}]}'
            if "西湖" in latest_user_message or "知识库" in latest_user_message:
                plan_json = '{"primary_category": "knowledge_retrieval", "steps": [{"executor": "rag"}, {"executor": "answer"}]}'
            elif "天气" in latest_user_message or "mcp" in latest_user_message.lower():
                plan_json = '{"primary_category": "mcp_tool_execution", "steps": [{"executor": "mcp"}, {"executor": "answer"}]}'
            elif "路况" in latest_user_message:
                plan_json = '{"primary_category": "traffic_status", "steps": [{"executor": "traffic"}, {"executor": "answer"}]}'
            elif "怎么走" in latest_user_message:
                plan_json = '{"primary_category": "route_planning", "steps": [{"executor": "rag"}, {"executor": "route"}, {"executor": "answer"}]}'
            elif "路网" in latest_user_message or "数据" in latest_user_message:
                plan_json = '{"primary_category": "report_generation", "steps": [{"executor": "report"}, {"executor": "answer"}]}'
            elif (
                "1+1" in latest_user_message
                or "计算" in latest_user_message
                or "时间" in latest_user_message
                or "几点" in latest_user_message
            ):
                plan_json = '{"primary_category": "general", "steps": [{"executor": "tool"}, {"executor": "answer"}]}'

            return AIMessage(
                content=plan_json,
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "total_tokens": 20,
                },
            )

        need_calculator_tool = "1+1" in latest_user_message or "计算" in latest_user_message
        if tools and not latest_tool_output and need_calculator_tool:
            return AIMessage(
                content="",
                response_metadata={
                    "finish_reason": "tool_calls",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                },
                tool_calls=[
                    {
                        "id": "call_calculator",
                        "name": "calculator",
                        "args": {"expression": "1+1"},
                        "type": "tool_call",
                    }
                ],
            )

        need_datetime_tool = "时间" in latest_user_message or "几点" in latest_user_message
        if tools and not latest_tool_output and need_datetime_tool:
            return AIMessage(
                content="",
                response_metadata={
                    "finish_reason": "tool_calls",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                },
                tool_calls=[
                    {
                        "id": "call_datetime",
                        "name": "current_datetime",
                        "args": {"timezone_name": "Asia/Shanghai"},
                        "type": "tool_call",
                    }
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
            return AIMessage(
                content="",
                response_metadata={
                    "finish_reason": "tool_calls",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                },
                tool_calls=[
                    {
                        "id": "call_mcp_weather",
                        "name": mcp_weather_tool_name,
                        "args": {"city": "杭州"},
                        "type": "tool_call",
                    }
                ],
            )

        if latest_tool_output:
            return AIMessage(
                content=f"测试模型回答：工具结果是 {latest_tool_output}",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                },
            )

        if "我刚刚告诉你的名字是什么" in latest_user_message:
            if history_contains_name:
                return AIMessage(
                    content="测试模型回答：你刚刚说你叫小王",
                    response_metadata={
                        "finish_reason": "stop",
                        "model_name": model_name or "test-model",
                    },
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                )
            if explicit_force_no_memory:
                return AIMessage(
                    content="测试模型回答：不知道",
                    response_metadata={
                        "finish_reason": "stop",
                        "model_name": model_name or "test-model",
                    },
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                )

        if has_knowledge_context and "西湖" in latest_user_message:
            return AIMessage(
                content="测试模型回答：根据知识库，西湖位于杭州。",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )
        if has_mcp_context and "MCP" in latest_user_message.upper():
            return AIMessage(
                content="测试模型回答：当前已配置 MCP 服务骨架。",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

        return AIMessage(
            content=f"测试模型回答：{latest_user_message}",
            response_metadata={"finish_reason": "stop", "model_name": model_name or "test-model"},
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )

    def fake_stream_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[AIMessageChunk]:
        """为流式集成测试返回稳定的增量结果。"""

        del self, tool_choice, enable_thinking
        latest_user_message = ""
        latest_tool_output = ""
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
            msg_type = _resolve_msg_type(message)
            content = str(getattr(message, "content", ""))
            if msg_type in ("tool", "function") and not latest_tool_output:
                latest_tool_output = content
            if msg_type == "human" and not latest_user_message:
                latest_user_message = content
            if latest_tool_output and latest_user_message:
                break

        for message in messages:
            msg_type = _resolve_msg_type(message)
            content = str(getattr(message, "content", ""))
            all_message_contents.append(content)
            if msg_type == "human":
                user_messages.append(content)

        async def iterator() -> AsyncIterator[AIMessageChunk]:
            resolved_model_name = model_name or "test-model"

            if latest_tool_output:
                full_text = f"测试模型回答：工具结果是 {latest_tool_output}"
            elif tools and ("1+1" in latest_user_message or "计算" in latest_user_message):
                yield AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "index": 0,
                            "id": "call_calculator",
                            "name": "calculator",
                            "args": '{"expression":"1+1"}',
                        }
                    ],
                )
                yield AIMessageChunk(
                    content="",
                    response_metadata={
                        "finish_reason": "tool_calls",
                        "model_name": resolved_model_name,
                    },
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                )
                return
            else:
                mcp_weather_tool_name = next(
                    (
                        tool_name
                        for tool_name in available_tool_names
                        if tool_name.startswith("mcp_") and "weather" in tool_name
                    ),
                    None,
                )
                if mcp_weather_tool_name:
                    yield AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "index": 0,
                                "id": "call_mcp_weather",
                                "name": mcp_weather_tool_name,
                                "args": '{"city":"杭州"}',
                            }
                        ],
                    )
                    yield AIMessageChunk(
                        content="",
                        response_metadata={
                            "finish_reason": "tool_calls",
                            "model_name": resolved_model_name,
                        },
                        usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
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
            yield AIMessageChunk(
                content=full_text[:split_index],
            )
            yield AIMessageChunk(
                content=full_text[split_index:],
            )
            yield AIMessageChunk(
                content="",
                response_metadata={"finish_reason": "stop", "model_name": resolved_model_name},
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

        return iterator()

    def fake_create_runnable(
        self: Any,
        *,
        model_name: str | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
        is_stream: bool = False,
    ) -> Any:
        """为集成测试返回假 Runnable，并确保它调用实例上的补全方法以便支持测试特定的 monkeypatch。"""
        llm_instance = self

        class FakeLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-llm"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                raise NotImplementedError("Use _agenerate instead")

            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                msg = await llm_instance.create_chat_completion(
                    messages=messages,
                    model_name=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    enable_thinking=enable_thinking,
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])

            async def _astream(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> AsyncIterator[ChatGenerationChunk]:
                # print(f"DEBUG: FakeLLM._astream called with {len(messages)} messages")
                async for chunk in llm_instance.stream_chat_completion(
                    messages=messages,
                    model_name=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    enable_thinking=enable_thinking,
                ):
                    yield ChatGenerationChunk(message=chunk)

        result = FakeLLM()
        result.name = "FakeLLM"
        return result

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.stream_chat_completion",
        fake_stream_chat_completion,
    )
    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fake_create_runnable,
    )

    from app.main import create_app

    application = create_app()

    with TestClient(application) as client:
        yield client
