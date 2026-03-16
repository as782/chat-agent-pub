"""MCP 管理器单元测试。"""

from __future__ import annotations

import json

import pytest
from pytest import MonkeyPatch

from app.mcp.manager import McpManager, McpServerDefinition


def test_mcp_manager_reads_server_definitions_from_environment(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器能够从环境变量读取服务器配置。"""

    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            [
                {
                    "name": "demo-http",
                    "transport": "http",
                    "endpoint": "https://mcp.example.com",
                }
            ],
            ensure_ascii=False,
        ),
    )

    manager = McpManager()
    server_list = manager.list_servers()

    assert server_list[0].name == "demo-http"
    assert server_list[0].transport == "http"
    assert server_list[0].endpoint == "https://mcp.example.com"


@pytest.mark.asyncio
async def test_mcp_manager_probes_http_server(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器会把 HTTP 服务器探测委托给 HTTP 客户端。"""

    async def fake_probe(self: object, endpoint: str) -> tuple[bool, str]:
        """返回稳定的 HTTP 探测结果。"""

        del self
        assert endpoint == "https://mcp.example.com"
        return True, "HTTP 服务可达"

    monkeypatch.setattr("app.mcp.http_client.McpHttpClient.probe", fake_probe)
    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="demo-http",
                transport="http",
                endpoint="https://mcp.example.com",
            )
        ]
    )

    probe_result = await manager.probe_server("demo-http")

    assert probe_result.is_available is True
    assert probe_result.detail == "HTTP 服务可达"


def test_mcp_manager_builds_agent_context() -> None:
    """验证 MCP 管理器会生成注入模型的服务说明。"""

    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="demo-http",
                transport="http",
                endpoint="https://mcp.example.com",
            ),
            McpServerDefinition(
                name="demo-stdio",
                transport="stdio",
                command="python",
                args=["server.py"],
            ),
        ]
    )

    mcp_context = manager.build_agent_context()

    assert mcp_context is not None
    assert "demo-http" in mcp_context
    assert "demo-stdio" in mcp_context
