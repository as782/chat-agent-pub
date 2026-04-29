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
    monkeypatch.setenv("LIVE_AGENT_BASE_URL", "http://localhost:8081")
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
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
        **kwargs: object,
    ) -> AIMessage:
        """为集成测试返回稳定的假模型结果。"""

        del self, api_key, base_url, timeout_seconds, tool_choice, enable_thinking, kwargs
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
        has_route_context = any("以下是当前路线规划问题" in message for message in all_message_contents)
        has_traffic_context = any("以下是当前路况类问题" in message for message in all_message_contents)
        has_service_context = any("以下是当前服务区查询结果" in message for message in all_message_contents)
        has_report_context = any("以下是当前路网报告任务" in message for message in all_message_contents)

        is_planner = any("生成分类与执行计划" in message for message in all_message_contents)
        if is_planner:
            plan_json = '{"primary_category": "general", "steps": [{"executor": "answer"}]}'
            if "西湖" in latest_user_message or "知识库" in latest_user_message:
                plan_json = '{"primary_category": "policy", "steps": [{"executor": "rag"}, {"executor": "answer"}]}'
            elif "天气" in latest_user_message or "mcp" in latest_user_message.lower():
                plan_json = '{"primary_category": "general", "steps": [{"executor": "mcp"}, {"executor": "answer"}]}'
            elif "服务区" in latest_user_message or "充电桩" in latest_user_message:
                plan_json = '{"primary_category": "service_area", "steps": [{"executor": "service"}, {"executor": "answer"}]}'
            elif "怎么走" in latest_user_message:
                plan_json = '{"primary_category": "route_planning", "steps": [{"executor": "route"}, {"executor": "answer"}]}'
            elif "路网" in latest_user_message or "数据" in latest_user_message:
                plan_json = '{"primary_category": "network_report", "steps": [{"executor": "report"}, {"executor": "answer"}]}'
            elif "路况" in latest_user_message:
                plan_json = '{"primary_category": "traffic_status", "steps": [{"executor": "traffic"}, {"executor": "answer"}]}'
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
        if has_route_context and "怎么走" in latest_user_message:
            return AIMessage(
                content="测试模型回答：推荐杭州到金华高速路线，约 2 小时。",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )
        if has_traffic_context and "路况" in latest_user_message:
            return AIMessage(
                content="测试模型回答：根据路况查询，杭州当前整体缓行，部分高架拥堵。",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )
        if has_service_context and ("服务区" in latest_user_message or "充电桩" in latest_user_message):
            return AIMessage(
                content="测试模型回答：杭州东服务区可提供充电和便利店服务，当前较繁忙。",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": model_name or "test-model",
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )
        if has_report_context and ("路网" in latest_user_message or "表格" in latest_user_message):
            return AIMessage(
                content="测试模型回答：全路网整体运行平稳，北向略有缓行。",
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
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[AIMessageChunk]:
        """为流式集成测试返回稳定的增量结果。"""

        del self, api_key, base_url, timeout_seconds, tool_choice, enable_thinking
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
            elif any("以下是当前路线规划问题" in message for message in all_message_contents):
                full_text = "测试模型回答：推荐杭州到金华高速路线，约 2 小时。"
            elif any("以下是当前路况类问题" in message for message in all_message_contents):
                full_text = "测试模型回答：根据路况查询，杭州当前整体缓行，部分高架拥堵。"
            elif any("以下是当前服务区查询结果" in message for message in all_message_contents):
                full_text = "测试模型回答：杭州东服务区可提供充电和便利店服务，当前较繁忙。"
            elif any("以下是当前路网报告任务" in message for message in all_message_contents):
                full_text = "测试模型回答：全路网整体运行平稳，北向略有缓行。"
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
        messages: list[object],
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
        is_stream: bool = False,
        **kwargs: object,
    ) -> Any:
        """为集成测试返回假 Runnable，并确保它调用实例上的补全方法以便支持测试特定的 monkeypatch。"""
        del messages, api_key, base_url, timeout_seconds, is_stream, kwargs
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
    async def fake_live_agent_request(
        self: object,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> object:
        """为直播问答接口客户端返回稳定的假数据。"""

        del self, method
        normalized_params = params or {}
        if path == "/agent/driving":
            return {
                "routesCount": 1,
                "routes": [
                    {
                        "distance": 180000,
                        "duration": 120,
                        "toll": 85,
                        "sections": [
                            {
                                "roadName": "沪昆高速",
                                "trafficControls": [],
                                "serviceAreas": [{"serviceName": "诸暨服务区"}],
                            }
                        ],
                    }
                ],
            }
        if path == "/agent/event":
            return [
                {
                    "roadName": str(normalized_params.get("road") or "杭州"),
                    "congestionInfoList": [{"id": "cg-1"}],
                    "trafficControlList": [{"id": "tc-1"}],
                    "serviceAreaList": [{"serviceName": "杭州服务区"}],
                    "exitInfoList": [{"tollName": "杭州南"}],
                }
            ]
        if path == "/agent/service":
            return [
                {
                    "serviceName": "杭州东服务区",
                    "roadName": "沪昆高速",
                    "statusTag": "繁忙",
                    "chargeList": [{"manufacturerName": "国网"}],
                    "commercialList": [{"name": "便利店"}],
                    "tags": ["餐饮", "休息区"],
                }
            ]
        if path == "/agent/topN":
            return {
                "queryTime": "2026-03-31 09:00:00",
                "congestion": {"totalMile": 12.5},
                "congestionTopN": [{"id": "cg-1", "roadName": "沪昆高速"}],
                "accidentTopN": [{"id": "ac-1", "roadName": "杭州绕城高速"}],
                "controlTopN": [{"id": "ct-1", "roadName": "长深高速"}],
            }
        raise AssertionError(f"unexpected live agent path: {path}")

    monkeypatch.setattr(
        "app.tools.live_agent.client.LiveAgentClient.request",
        fake_live_agent_request,
    )

    from app.main import create_app

    application = create_app()

    with TestClient(application) as client:
        yield client


@pytest.fixture
def regression_trace() -> dict[str, list[object]]:
    """记录接口级回归测试中 live-agent 与知识库链路的调用轨迹。"""

    return {
        "live_agent_calls": [],
        "rag_queries": [],
    }


@pytest.fixture
def regression_app_client(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    regression_trace: dict[str, list[object]],
) -> Iterator[TestClient]:
    """提供面向真实回归问题的接口级测试客户端。"""

    sqlite_database_path = tmp_path / "regression-test.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("POSTGRES_DSN", f"sqlite+aiosqlite:///{sqlite_database_path.as_posix()}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("RAGFLOW_API_KEY", "test-ragflow-key")
    monkeypatch.setenv("RAGFLOW_BASE_URL", "https://ragflow.example.com")
    monkeypatch.setenv("LIVE_AGENT_BASE_URL", "http://localhost:8081")

    def _resolve_msg_type(message: object) -> str:
        msg_type = getattr(message, "type", None)
        if msg_type:
            return str(msg_type)
        role = getattr(message, "role", "")
        if role == "user":
            return "human"
        if role == "assistant":
            return "ai"
        return str(role)

    def _match_regression_scenario(message: str) -> str:
        normalized = str(message).replace("？", "?").strip()
        if (
            "收割机" in normalized
            and "杭州" in normalized
            and "宁波" in normalized
            and any(keyword in normalized for keyword in ("费用", "收费", "多少钱", "多少"))
        ):
            return "harvester_fee"
        if (
            "农机" in normalized
            and "绍兴" in normalized
            and "宁波" in normalized
            and any(keyword in normalized for keyword in ("免费", "规定", "要求", "政策"))
        ):
            return "agri_policy"
        if "杭州" in normalized and "宁波" in normalized and "堵不堵" in normalized:
            return "hz_nb_congestion"
        if (
            "杭州" in normalized
            and "宁波" in normalized
            and any(keyword in normalized for keyword in ("推荐一下路线", "推荐路线", "推荐一下", "路线"))
        ):
            return "hz_nb_route"
        if (
            "杭州" in normalized
            and "宁波" in normalized
            and any(keyword in normalized for keyword in ("收费多少", "多少钱", "过路费", "通行费"))
        ):
            return "hz_nb_toll"
        if "甲鱼" in normalized and any(keyword in normalized for keyword in ("免费", "绿通")):
            return "softshell_turtle_free"
        return "default"

    def _build_planner_response(scenario: str, model_name: str | None) -> AIMessage:
        if scenario == "harvester_fee":
            plan_json = json.dumps(
                {
                    "primary_category": "route_planning",
                    "steps": [
                        {"executor": "route"},
                        {"executor": "rag"},
                        {
                            "executor": "answer",
                            "metadata": {"focus": "政策资格判断与路线收费说明"},
                        },
                    ],
                },
                ensure_ascii=False,
            )
        elif scenario == "agri_policy":
            plan_json = json.dumps(
                {
                    "primary_category": "policy",
                    "steps": [
                        {"executor": "rag"},
                        {"executor": "route"},
                        {
                            "executor": "answer",
                            "metadata": {"focus": "政策资格判断与路线收费说明"},
                        },
                    ],
                },
                ensure_ascii=False,
            )
        elif scenario in {"hz_nb_congestion", "hz_nb_route", "hz_nb_toll"}:
            plan_json = json.dumps(
                {
                    "primary_category": "route_planning",
                    "steps": [{"executor": "route"}, {"executor": "answer"}],
                },
                ensure_ascii=False,
            )
        elif scenario == "softshell_turtle_free":
            plan_json = json.dumps(
                {
                    "primary_category": "policy",
                    "steps": [{"executor": "rag"}, {"executor": "answer"}],
                },
                ensure_ascii=False,
            )
        else:
            plan_json = json.dumps(
                {
                    "primary_category": "general",
                    "steps": [{"executor": "answer"}],
                },
                ensure_ascii=False,
            )
        return AIMessage(
            content=plan_json,
            response_metadata={
                "finish_reason": "stop",
                "model_name": model_name or "test-model",
            },
            usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        )

    def _build_answer_text(scenario: str, *, has_route_context: bool, has_knowledge_context: bool) -> str:
        if scenario == "harvester_fee":
            return (
                "联合收割机跨区作业车辆如果属于《1.014关于切实做好联合收割机(插秧机)跨区作业通行"
                "服务保障工作的通知》适用范围，并已按要求办理作业证、预约等手续，通行费可以免收。"
                "当前路线工具返回的64至92元只能作为普通车辆收费参考，若证件或装载条件未核实，不能直接按普通收费下结论。"
            )
        if scenario == "agri_policy":
            return (
                "农机跨区作业是否免费，要先看车辆是否属于政策适用对象，以及作业证、预约信息、车辆与作业任务"
                "是否一致。满足条件时可免费通行；不满足条件时，再参考绍兴到宁波普通路线收费。"
            )
        if scenario == "hz_nb_congestion":
            return "杭州到宁波当前整体通行基本畅通，没有明显拥堵和管制，走常规高速路线即可。"
        if scenario == "hz_nb_toll":
            return "普通小客车从杭州到宁波的高速费参考约64至92元，较常用路线大约65元。"
        if scenario == "hz_nb_route":
            return "推荐优先走杭州湾环线高速一线，整体更直接，预计约2小时左右可到宁波。"
        if scenario == "softshell_turtle_free":
            return (
                "甲鱼不能享受鲜活农产品运输绿色通道免收通行费政策。"
                "原因是甲鱼不在全国统一《鲜活农产品品种目录》范围内，因此应按普通货运车辆规则缴纳通行费。"
            )
        return "测试模型回答：默认回复"

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
        **kwargs: object,
    ) -> AIMessage:
        del self, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking, kwargs
        latest_user_message = ""
        all_message_contents: list[str] = []

        for message in reversed(messages):
            msg_type = _resolve_msg_type(message)
            content = str(getattr(message, "content", ""))
            all_message_contents.append(content)
            if msg_type == "human" and not latest_user_message:
                latest_user_message = content

        is_planner = any("生成分类与执行计划" in message for message in all_message_contents)
        scenario = _match_regression_scenario(latest_user_message)
        if is_planner:
            return _build_planner_response(scenario, model_name)

        has_knowledge_context = any("以下是知识库检索结果" in message for message in all_message_contents)
        has_route_context = any("以下是当前路线规划问题" in message for message in all_message_contents)
        return AIMessage(
            content=_build_answer_text(
                scenario,
                has_route_context=has_route_context,
                has_knowledge_context=has_knowledge_context,
            ),
            response_metadata={
                "finish_reason": "stop",
                "model_name": model_name or "test-model",
            },
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )

    def fake_stream_chat_completion(
        self: object,
        messages: list[object],
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[AIMessageChunk]:
        del self, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking

        latest_user_message = ""
        all_message_contents: list[str] = []
        for message in reversed(messages):
            msg_type = _resolve_msg_type(message)
            content = str(getattr(message, "content", ""))
            all_message_contents.append(content)
            if msg_type == "human" and not latest_user_message:
                latest_user_message = content

        scenario = _match_regression_scenario(latest_user_message)
        has_knowledge_context = any("以下是知识库检索结果" in message for message in all_message_contents)
        has_route_context = any("以下是当前路线规划问题" in message for message in all_message_contents)
        full_text = _build_answer_text(
            scenario,
            has_route_context=has_route_context,
            has_knowledge_context=has_knowledge_context,
        )

        async def iterator() -> AsyncIterator[AIMessageChunk]:
            resolved_model_name = model_name or "test-model"
            split_index = max(1, len(full_text) // 2)
            yield AIMessageChunk(content=full_text[:split_index])
            yield AIMessageChunk(content=full_text[split_index:])
            yield AIMessageChunk(
                content="",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": resolved_model_name,
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

        return iterator()

    def fake_create_runnable(
        self: Any,
        *,
        messages: list[object],
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        tools: list[object] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        enable_thinking: bool | None = None,
        is_stream: bool = False,
        **kwargs: object,
    ) -> Any:
        del messages, api_key, base_url, timeout_seconds, is_stream, kwargs
        llm_instance = self

        class FakeLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-regression-llm"

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
                del stop, run_manager, kwargs
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
                del stop, run_manager, kwargs
                async for chunk in llm_instance.stream_chat_completion(
                    messages=messages,
                    model_name=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    enable_thinking=enable_thinking,
                ):
                    yield ChatGenerationChunk(message=chunk)

        result = FakeLLM()
        result.name = "FakeRegressionLLM"
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

    async def fake_live_agent_request(
        self: object,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> object:
        del self, method
        regression_trace["live_agent_calls"].append(
            {
                "path": path,
                "params": dict(params or {}),
            }
        )
        if path != "/agent/driving":
            raise AssertionError(f"unexpected live agent path: {path}")
        return {
            "routesCount": 3,
            "routes": [
                {
                    "distance": 179000,
                    "duration": 120,
                    "toll": 65,
                    "sections": [{"roadName": "杭州湾环线高速", "trafficControls": []}],
                },
                {
                    "distance": 177000,
                    "duration": 123,
                    "toll": 64,
                    "sections": [{"roadName": "杭甬高速", "trafficControls": []}],
                },
                {
                    "distance": 190000,
                    "duration": 136,
                    "toll": 92,
                    "sections": [{"roadName": "常规绕行路线", "trafficControls": []}],
                },
            ],
        }

    monkeypatch.setattr(
        "app.tools.live_agent.client.LiveAgentClient.request",
        fake_live_agent_request,
    )

    from app.schemas.knowledge import KnowledgeSearchResult

    async def fake_retrieve_for_agent(
        self: object,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        del self, top_k
        regression_trace["rag_queries"].append(query)
        if any(keyword in query for keyword in ("收割机", "联合收割机", "农机", "插秧机")):
            return [
                KnowledgeSearchResult(
                    document_id="doc-harvester-001",
                    chunk_id=f"chunk-{len(regression_trace['rag_queries'])}",
                    score=0.97,
                    content=(
                        "根据《关于切实做好联合收割机(插秧机)跨区作业通行服务保障工作的通知》，"
                        "对符合跨区作业条件并按要求办理作业证、预约等手续的联合收割机(插秧机)"
                        "运输车辆，免收车辆通行费。"
                    ),
                    source="1.014关于切实做好联合收割机(插秧机)跨区作业通行服务保障工作的通知",
                )
            ]
        if "甲鱼" in query:
            return [
                KnowledgeSearchResult(
                    document_id="doc-softshell-turtle-001",
                    chunk_id=f"chunk-{len(regression_trace['rag_queries'])}",
                    score=0.96,
                    content=(
                        "甲鱼不在全国统一《鲜活农产品品种目录》范围内，"
                        "因此运输甲鱼的车辆不能享受鲜活农产品绿色通道免收通行费政策。"
                    ),
                    source="浙江场景问答-甲鱼绿通口径",
                )
            ]
        return []

    monkeypatch.setattr(
        "app.agent.nodes.ragflow_node.KnowledgeService.retrieve_for_agent",
        fake_retrieve_for_agent,
    )

    from app.main import create_app

    application = create_app()
    with TestClient(application) as client:
        yield client
