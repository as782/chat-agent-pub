"""MCP 接口集成测试。"""

from fastapi.testclient import TestClient

from app.schemas.mcp import McpToolCallResponse, McpToolInfo


def test_mcp_api_lists_servers(app_client: TestClient, monkeypatch) -> None:
    """验证 MCP 接口可以返回服务列表。"""

    def fake_list_servers(self: object) -> list[object]:
        """返回稳定的 MCP 服务列表。"""

        del self
        from app.schemas.mcp import McpServerInfo

        return [
            McpServerInfo(
                name="demo-http",
                transport="http",
                endpoint="https://mcp.example.com/mcp",
            )
        ]

    monkeypatch.setattr("app.mcp.manager.McpManager.list_servers", fake_list_servers)

    response = app_client.get("/api/v1/mcp/servers")

    assert response.status_code == 200
    assert response.json()["items"][0]["name"] == "demo-http"


def test_mcp_api_probes_server(app_client: TestClient, monkeypatch) -> None:
    """验证 MCP 接口可以返回服务探测结果。"""

    async def fake_probe_server(self: object, server_name: str):
        """返回稳定的探测结果。"""

        del self
        from app.schemas.mcp import McpProbeResponse

        assert server_name == "demo-http"
        return McpProbeResponse(
            name="demo-http",
            transport="http",
            is_available=True,
            detail="HTTP MCP 服务可用，共发现 1 个工具",
        )

    monkeypatch.setattr("app.mcp.manager.McpManager.probe_server", fake_probe_server)

    response = app_client.post("/api/v1/mcp/servers/demo-http/probe")

    assert response.status_code == 200
    assert response.json()["is_available"] is True


def test_mcp_api_lists_server_tools(app_client: TestClient, monkeypatch) -> None:
    """验证 MCP 接口可以列出指定服务的工具。"""

    async def fake_list_tools(self: object, server_name: str) -> list[McpToolInfo]:
        """返回稳定的工具列表。"""

        del self
        assert server_name == "demo-http"
        return [
            McpToolInfo(
                server_name="demo-http",
                name="weather",
                registered_name="mcp_demo_http__weather",
                description="查询天气。",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ]

    monkeypatch.setattr("app.mcp.manager.McpManager.list_tools", fake_list_tools)

    response = app_client.get("/api/v1/mcp/servers/demo-http/tools")

    assert response.status_code == 200
    assert response.json()["items"][0]["registered_name"] == "mcp_demo_http__weather"


def test_mcp_api_calls_tool(app_client: TestClient, monkeypatch) -> None:
    """验证 MCP 接口可以调用指定工具。"""

    async def fake_call_tool(
        self: object,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpToolCallResponse:
        """返回稳定的工具调用结果。"""

        del self
        assert server_name == "demo-http"
        assert tool_name == "weather"
        assert arguments == {"city": "杭州"}
        return McpToolCallResponse(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
            content=[{"type": "text", "text": "杭州晴，26 度"}],
            structured_content={"city": "杭州"},
            is_error=False,
            output_text="杭州晴，26 度",
        )

    monkeypatch.setattr("app.mcp.manager.McpManager.call_tool", fake_call_tool)

    response = app_client.post(
        "/api/v1/mcp/servers/demo-http/tools/weather/call",
        json={"arguments": {"city": "杭州"}},
    )

    assert response.status_code == 200
    assert response.json()["output_text"] == "杭州晴，26 度"


def test_mcp_api_returns_400_when_server_not_found(app_client: TestClient) -> None:
    """验证探测不存在的 MCP 服务时会返回明确错误。"""

    response = app_client.post("/api/v1/mcp/servers/not-found/probe")

    assert response.status_code == 400
    assert response.json()["error_code"] == "mcp_server_not_found"
