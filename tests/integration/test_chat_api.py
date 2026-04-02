"""对话接口集成测试。"""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.core.exceptions import UpstreamServiceException
from app.mcp.models import McpRuntimeTool
from app.schemas.knowledge import KnowledgeSearchResult


def test_chat_api_creates_session_and_returns_answer(app_client: TestClient) -> None:
    """验证内部聊天接口会返回 OpenAI 兼容响应，并通过响应头暴露会话标识。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好，系统"}],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]

    assert response.status_code == 200
    assert session_id
    assert response_payload["id"].startswith("chatcmpl-")
    assert response_payload["object"] == "chat.completion"
    assert response_payload["model"] == "test-model"
    assert response_payload["choices"][0]["message"]["role"] == "assistant"
    assert response_payload["choices"][0]["message"]["content"] == "测试模型回答：你好，系统"
    assert response_payload["choices"][0]["finish_reason"] == "stop"
    assert response_payload["usage"]["total_tokens"] == 20


def test_chat_api_executes_builtin_tool_when_enabled(app_client: TestClient) -> None:
    """验证内部聊天接口会在后台执行工具，但对外仍返回 OpenAI 兼容格式。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "请帮我计算 1+1"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                }
            ],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]

    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert response_payload["choices"][0]["message"]["content"] == "测试模型回答：工具结果是 2"
    assert response_payload["choices"][0]["finish_reason"] == "stop"
    assert history_response.status_code == 200
    assert history_payload["total"] == 4
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history_payload["items"][1]["metadata"]["tool_calls"][0]["tool_name"] == "calculator"
    assert history_payload["items"][2]["content"] == "2"


def test_chat_api_supports_multi_turn_memory(app_client: TestClient) -> None:
    """验证同一会话下会自动注入历史消息，实现最小多轮记忆。"""

    first_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )
    session_id = first_response.headers["X-Session-ID"]

    second_response = app_client.post(
        "/api/v1/chat",
        headers={"X-Session-ID": session_id},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我刚刚告诉你的名字是什么？"}],
        },
    )

    assert second_response.status_code == 200
    assert (
        second_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )


def test_chat_api_combines_session_memory_and_explicit_messages(app_client: TestClient) -> None:
    """验证带 session_id 时，会结合系统历史和本次显式 messages 回答。"""

    first_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )
    session_id = first_response.headers["X-Session-ID"]

    second_response = app_client.post(
        "/api/v1/chat",
        headers={"X-Session-ID": session_id},
        json={
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": "请结合系统记录和本次输入回答。",
                },
                {"role": "user", "content": "我刚刚告诉你的名字是什么？"},
            ],
        },
    )

    assert second_response.status_code == 200
    assert (
        second_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )


def test_chat_api_does_not_use_other_session_memory_without_session_id(
    app_client: TestClient,
) -> None:
    """验证不携带 session_id 时，不会读取其他会话已保存的系统记忆。"""

    first_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )

    second_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "我叫小王，请记住这个名字。"},
                {"role": "assistant", "content": "好的，我记住了。"},
                {"role": "user", "content": "我刚刚告诉你的名字是什么？"},
            ],
        },
    )

    assert second_response.status_code == 200
    assert (
        second_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )
    assert second_response.headers["X-Session-ID"] != first_response.headers["X-Session-ID"]


def test_chat_api_uses_knowledge_route_when_requested(app_client: TestClient, monkeypatch) -> None:
    """验证命中知识库路由时会把检索结果注入回答上下文。"""

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

    monkeypatch.setattr(
        "app.agent.nodes.ragflow_node.KnowledgeService.retrieve_for_agent",
        fake_retrieve_for_agent,
    )

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "知识库: 西湖在哪里？"}],
        },
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：根据知识库，西湖位于杭州。"
    )


def test_chat_api_executes_route_query_via_live_tools(app_client: TestClient) -> None:
    """验证路线规划问题会直接查询标准工具并返回汇总回答。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "杭州到金华怎么走？"}],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]
    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert response_payload["choices"][0]["message"]["content"] == (
        "测试模型回答：推荐杭州到金华高速路线，约 2 小时。"
    )
    assert [message["role"] for message in history_payload["items"]] == ["user", "assistant"]


def test_chat_api_uses_misspelled_knowledge_prefix_when_requested(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证 konwledge 前缀也会命中知识库分支。"""

    async def fake_retrieve_for_agent(
        self: object,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        """返回稳定的知识检索结果。"""

        del self, top_k
        assert query == "高速清障最低标准是什么？"
        return [
            KnowledgeSearchResult(
                document_id="doc-002",
                chunk_id="chunk-002",
                score=0.95,
                content="高速清障最低标准以实际法规和运营单位要求为准。",
                source="高速运营规范",
            )
        ]

    monkeypatch.setattr(
        "app.agent.nodes.ragflow_node.KnowledgeService.retrieve_for_agent",
        fake_retrieve_for_agent,
    )

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "konwledge: 高速清障最低标准是什么？"}],
        },
    )

    assert response.status_code == 200


def test_chat_api_executes_mcp_tool_when_requested(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证命中 MCP 路由时会拉取 MCP 工具并完成真实工具调用闭环。"""

    async def fake_build_runtime_tools(self: object, *, server_names=None) -> list[McpRuntimeTool]:
        """返回稳定的 MCP 运行时工具集合。"""

        del self, server_names
        return [
            McpRuntimeTool(
                registered_name="mcp_demo_http__weather",
                server_name="demo-mcp-http",
                remote_tool_name="weather",
                description="查询城市天气。",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ]

    async def fake_call_tool(
        self: object,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
    ):
        """返回稳定的 MCP 工具调用结果。"""

        del self
        assert server_name == "demo-mcp-http"
        assert tool_name == "weather"
        assert arguments == {"city": "杭州"}
        from app.schemas.mcp import McpToolCallResponse

        return McpToolCallResponse(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
            content=[{"type": "text", "text": "Hangzhou sunny, 26 C"}],
            structured_content={"city": "Hangzhou", "weather": "sunny", "temperature": "26"},
            is_error=False,
            output_text="Hangzhou sunny, 26 C",
        )

    monkeypatch.setattr(
        "app.mcp.manager.McpManager.build_runtime_tools",
        fake_build_runtime_tools,
    )
    monkeypatch.setattr("app.mcp.manager.McpManager.call_tool", fake_call_tool)

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "mcp: 帮我查杭州天气"}],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]
    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert (
        response_payload["choices"][0]["message"]["content"]
        == "测试模型回答：工具结果是 Hangzhou sunny, 26 C"
    )
    assert history_response.status_code == 200
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history_payload["items"][2]["metadata"]["tool_name"] == "mcp_demo_http__weather"


def test_chat_api_executes_traffic_query_via_live_tools(app_client: TestClient) -> None:
    """验证路况问题会直接查询标准工具并返回汇总回答。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "当前杭州路况怎么样？"}],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]
    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert response_payload["choices"][0]["message"]["content"] == (
        "测试模型回答：根据路况查询，杭州当前整体缓行，部分高架拥堵。"
    )
    assert [message["role"] for message in history_payload["items"]] == ["user", "assistant"]


def test_chat_api_executes_service_query_via_live_tools(app_client: TestClient) -> None:
    """验证服务区问题会直接查询标准工具并返回汇总回答。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "杭州东服务区充电桩情况怎么样？"}],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]
    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert response_payload["choices"][0]["message"]["content"] == (
        "测试模型回答：杭州东服务区可提供充电和便利店服务，当前较繁忙。"
    )
    assert [message["role"] for message in history_payload["items"]] == ["user", "assistant"]


def test_chat_api_executes_report_query_via_live_tools(app_client: TestClient) -> None:
    """验证整体路网问题会直接查询标准工具并返回汇总回答。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "请生成今天全路网路况对比表格"}],
        },
    )
    response_payload = response.json()
    session_id = response.headers["X-Session-ID"]
    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert response_payload["choices"][0]["message"]["content"] == (
        "测试模型回答：全路网整体运行平稳，北向略有缓行。"
    )
    assert [message["role"] for message in history_payload["items"]] == ["user", "assistant"]


def test_chat_api_returns_json_error_when_live_agent_upstream_fails(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证上游接口失败时，非流式接口会直接返回错误响应。"""

    async def fake_live_agent_request(
        self: object,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> object:
        """模拟直播问答接口上游失败。"""

        del self, method, path, params
        raise UpstreamServiceException(
            "直播问答接口异常。",
            error_code="live_agent_connection_error",
        )

    monkeypatch.setattr(
        "app.tools.live_agent.client.LiveAgentClient.request",
        fake_live_agent_request,
    )

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "杭州到金华怎么走？"}],
        },
    )
    response_payload = response.json()

    assert response.status_code == 503
    assert response_payload["error_code"] == "live_agent_connection_error"
    assert response_payload["message"] == "上游接口报错，请稍后重试。"


def test_chat_api_streams_response_when_requested(app_client: TestClient) -> None:
    """验证内部聊天接口在 stream=true 时返回 OpenAI 兼容 SSE 数据。"""

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["X-Session-ID"]
    assert '"object": "chat.completion.chunk"' in response_body
    assert response_body.count('"role": "assistant"') >= 2
    assert response_body.count('"content":') >= 2
    assert "[DONE]" in response_body


def test_chat_api_stream_executes_builtin_tool_when_requested(app_client: TestClient) -> None:
    """验证流式内部聊天接口会在同一条 SSE 中完成工具调用后二阶段续答。"""

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "请帮我计算 1+1"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                }
            ],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")
        session_id = response.headers["X-Session-ID"]

    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert '"tool_calls"' in response_body
    assert "测试模型回答：" in response_body
    assert "工具结果是 2" in response_body
    assert "[DONE]" in response_body
    assert history_response.status_code == 200
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history_payload["items"][2]["content"] == "2"
    assert history_payload["items"][3]["content"] == "测试模型回答：工具结果是 2"


def test_chat_api_stream_hides_tool_node_reasoning_text(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证流式输出不会透传 tool_node 的中间规划文本，只保留工具调用与最终回答。"""

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
        del self, api_key, base_url, timeout_seconds, tool_choice, enable_thinking
        latest_user_message = ""
        latest_tool_output = ""

        for message in reversed(messages):
            msg_type = getattr(message, "type", None) or getattr(message, "role", "")
            content = str(getattr(message, "content", ""))
            if msg_type in ("tool", "function") and not latest_tool_output:
                latest_tool_output = content
            if msg_type in ("human", "user") and not latest_user_message:
                latest_user_message = content
            if latest_tool_output and latest_user_message:
                break

        async def iterator() -> AsyncIterator[AIMessageChunk]:
            resolved_model_name = model_name or "test-model"

            if tools and not latest_tool_output and ("1+1" in latest_user_message):
                # 模拟模型在工具调用前输出“规划 JSON + 解释文本”。
                yield AIMessageChunk(
                    content=(
                        '{"primary_category":"route_planning","steps":[{"executor":"tool"}]}'
                    )
                )
                yield AIMessageChunk(content="先调用工具获取结果。")
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

            full_text = f"测试模型回答：工具结果是 {latest_tool_output}"
            yield AIMessageChunk(content=full_text)
            yield AIMessageChunk(
                content="",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": resolved_model_name,
                },
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

        return iterator()

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.stream_chat_completion",
        fake_stream_chat_completion,
    )

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "请帮我计算 1+1"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                }
            ],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert '"tool_calls"' in response_body
    assert "测试模型回答：工具结果是 2" in response_body
    assert "primary_category" not in response_body
    assert "先调用工具获取结果" not in response_body
    assert response_body.count("[DONE]") == 1


def test_chat_api_stream_executes_mcp_tool_when_requested(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证流式内部聊天接口会在同一条 SSE 中完成 MCP 工具执行后二阶段续答。"""

    async def fake_build_runtime_tools(self: object, *, server_names=None) -> list[McpRuntimeTool]:
        """返回稳定的 MCP 运行时工具集合。"""

        del self, server_names
        return [
            McpRuntimeTool(
                registered_name="mcp_demo_http__weather",
                server_name="demo-mcp-http",
                remote_tool_name="weather",
                description="查询城市天气。",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ]

    async def fake_call_tool(
        self: object,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
    ):
        """返回稳定的 MCP 工具调用结果。"""

        del self
        assert server_name == "demo-mcp-http"
        assert tool_name == "weather"
        assert arguments == {"city": "杭州"}
        from app.schemas.mcp import McpToolCallResponse

        return McpToolCallResponse(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
            content=[{"type": "text", "text": "Hangzhou sunny, 26 C"}],
            structured_content={"city": "Hangzhou", "weather": "sunny", "temperature": "26"},
            is_error=False,
            output_text="Hangzhou sunny, 26 C",
        )

    monkeypatch.setattr(
        "app.mcp.manager.McpManager.build_runtime_tools",
        fake_build_runtime_tools,
    )
    monkeypatch.setattr("app.mcp.manager.McpManager.call_tool", fake_call_tool)

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "mcp: 帮我查杭州天气"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")
        session_id = response.headers["X-Session-ID"]

    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert '"tool_calls"' in response_body
    assert "测试模型回答：" in response_body
    assert "工具结果是" in response_body
    assert "[DONE]" in response_body
    assert history_response.status_code == 200
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history_payload["items"][2]["metadata"]["tool_name"] == "mcp_demo_http__weather"
    assert history_payload["items"][3]["content"] == "测试模型回答：工具结果是 Hangzhou sunny, 26 C"


def test_chat_api_stream_executes_route_query_via_live_tools(app_client: TestClient) -> None:
    """验证流式路线问题会直接查询标准工具并输出最终回答。"""

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "杭州到金华怎么走？"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")
        session_id = response.headers["X-Session-ID"]

    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert "[DONE]" in response_body
    assert [message["role"] for message in history_payload["items"]] == ["user", "assistant"]
    assert (
        history_payload["items"][1]["content"]
        == "测试模型回答：推荐杭州到金华高速路线，约 2 小时。"
    )


def test_chat_api_stream_executes_traffic_query_via_live_tools(app_client: TestClient) -> None:
    """验证流式路况问题会直接查询标准工具并输出最终回答。"""

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "当前杭州路况怎么样？"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")
        session_id = response.headers["X-Session-ID"]

    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert response.status_code == 200
    assert "[DONE]" in response_body
    assert [message["role"] for message in history_payload["items"]] == ["user", "assistant"]
    assert (
        history_payload["items"][1]["content"]
        == "测试模型回答：根据路况查询，杭州当前整体缓行，部分高架拥堵。"
    )


def test_chat_api_stream_returns_json_error_when_knowledge_upstream_fails(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证上游接口失败时，流式接口在首包前直接返回 JSON 错误。"""

    async def fake_retrieve_for_agent(
        self: object,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        """模拟知识库上游失败。"""

        del self, query, top_k
        raise UpstreamServiceException(
            "RAGFlow 连接失败。",
            error_code="ragflow_connection_error",
        )

    monkeypatch.setattr(
        "app.agent.nodes.ragflow_node.KnowledgeService.retrieve_for_agent",
        fake_retrieve_for_agent,
    )

    with app_client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "知识库: 西湖在哪里？"}],
            "stream": True,
        },
    ) as response:
        response_body = response.read().decode("utf-8")

    assert response.status_code == 503
    assert '"error_code":"ragflow_connection_error"' in response_body.replace(" ", "")
    assert "上游接口报错，请稍后重试。" in response_body


def test_chat_api_returns_404_when_session_not_found(app_client: TestClient) -> None:
    """验证对话接口在会话不存在时返回 404。"""

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "继续回答"}],
        },
        headers={"X-Session-ID": "not-exists"},
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "resource_not_found"


def test_chat_api_stream_returns_json_error_when_first_chunk_fails(
    app_client: TestClient,
    monkeypatch,
) -> None:
    """验证内部流式接口在首块失败时返回 JSON 错误，而不是直接中断连接。"""

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
        """模拟首个流式块生成前发生限流错误。"""

        del self, messages, model_name, api_key, base_url, timeout_seconds, tools, tool_choice, enable_thinking

        async def iterator() -> AsyncIterator[AIMessageChunk]:
            raise UpstreamServiceException(
                "LLM 提供方触发限流，请稍后重试。",
                error_code="llm_rate_limited",
                status_code=429,
            )
            yield AIMessageChunk(content="")

        return iterator()

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.stream_chat_completion",
        fake_stream_chat_completion,
    )

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    )

    assert response.status_code == 429
    assert response.json()["error_code"] == "llm_rate_limited"
