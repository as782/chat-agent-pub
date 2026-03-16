"""MCP 接口集成测试。"""

from fastapi.testclient import TestClient


def test_mcp_api_lists_servers(app_client: TestClient, monkeypatch) -> None:
    """验证 MCP 接口可以返回服务器列表。"""

    def fake_list_servers(self: object) -> list[object]:
        """返回稳定的 MCP 服务器列表。"""

        del self
        from app.schemas.mcp import McpServerInfo

        return [
            McpServerInfo(
                name="demo-http",
                transport="http",
                endpoint="https://mcp.example.com",
            )
        ]

    monkeypatch.setattr("app.mcp.manager.McpManager.list_servers", fake_list_servers)

    response = app_client.get("/api/v1/mcp/servers")

    assert response.status_code == 200
    assert response.json()["items"][0]["name"] == "demo-http"


def test_mcp_api_probes_server(app_client: TestClient, monkeypatch) -> None:
    """验证 MCP 接口可以返回服务器探测结果。"""

    async def fake_probe_server(self: object, server_name: str):
        """返回稳定的探测结果。"""

        del self
        from app.schemas.mcp import McpProbeResponse

        assert server_name == "demo-http"
        return McpProbeResponse(
            name="demo-http",
            transport="http",
            is_available=True,
            detail="HTTP 服务可达",
        )

    monkeypatch.setattr("app.mcp.manager.McpManager.probe_server", fake_probe_server)

    response = app_client.post("/api/v1/mcp/servers/demo-http/probe")

    assert response.status_code == 200
    assert response.json()["is_available"] is True


def test_mcp_api_returns_400_when_server_not_found(app_client: TestClient) -> None:
    """验证探测不存在的 MCP 服务时会返回明确错误。"""

    response = app_client.post("/api/v1/mcp/servers/not-found/probe")

    assert response.status_code == 400
    assert response.json()["error_code"] == "mcp_server_not_found"
