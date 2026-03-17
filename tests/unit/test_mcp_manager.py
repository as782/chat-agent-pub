"""MCP 管理器单元测试。"""

from __future__ import annotations

import json

import pytest
from pytest import MonkeyPatch

from app.mcp.manager import McpManager, McpServerDefinition
from app.mcp.models import McpClientToolCallResult, McpClientToolDefinition


def test_mcp_manager_reads_server_definitions_from_environment(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器能够从环境变量读取服务配置。"""

    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            [
                {
                    "name": "demo-http",
                    "transport": "streamable_http",
                    "endpoint": "https://mcp.example.com/mcp",
                }
            ],
            ensure_ascii=False,
        ),
    )

    manager = McpManager()
    server_list = manager.list_servers()

    assert server_list[0].name == "demo-http"
    assert server_list[0].transport == "streamable_http"
    assert server_list[0].endpoint == "https://mcp.example.com/mcp"


def test_mcp_manager_reads_amap_style_mapping_from_environment(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证 MCP 管理器兼容高德文档中的 mcpServers 映射格式。"""

    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            {
                "mcpServers": {
                    "amap-maps-streamableHTTP": {
                        "url": "https://mcp.amap.com/mcp?key=test-key",
                    }
                }
            },
            ensure_ascii=False,
        ),
    )

    manager = McpManager()
    server_list = manager.list_servers()

    assert server_list[0].name == "amap-maps-streamableHTTP"
    assert server_list[0].transport == "http"
    assert server_list[0].endpoint == "https://mcp.amap.com/mcp?key=test-key"


def test_mcp_manager_reads_list_wrapped_mcp_servers_mapping(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证 MCP 管理器兼容列表包裹的 mcpServers 映射格式。"""

    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            [
                {
                    "mcpServers": {
                        "amap": {
                            "url": "https://mcp.amap.com/mcp?key=test-key",
                            "headers": {"x-test": "1"},
                        }
                    }
                }
            ],
            ensure_ascii=False,
        ),
    )

    manager = McpManager()
    server_list = manager.list_servers()

    assert server_list[0].name == "amap"
    assert server_list[0].transport == "http"
    assert server_list[0].endpoint == "https://mcp.amap.com/mcp?key=test-key"


def test_mcp_manager_infers_sse_transport_from_url(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器能从常见 SSE URL 自动推断传输方式。"""

    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            {"mcpServers": {"demo-sse": {"url": "https://mcp.example.com/sse"}}},
            ensure_ascii=False,
        ),
    )

    manager = McpManager()
    server_list = manager.list_servers()

    assert server_list[0].name == "demo-sse"
    assert server_list[0].transport == "sse"
    assert server_list[0].endpoint == "https://mcp.example.com/sse"


@pytest.mark.asyncio
async def test_mcp_manager_probes_http_server(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器会把 HTTP 服务探测委托给 HTTP 客户端。"""

    async def fake_probe(
        self: object,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """返回稳定的 HTTP 探测结果。"""

        del self, headers
        assert endpoint == "https://mcp.example.com/mcp"
        return True, "HTTP MCP 服务可用，共发现 3 个工具"

    monkeypatch.setattr("app.mcp.http_client.McpHttpClient.probe", fake_probe)
    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="demo-http",
                transport="streamable_http",
                endpoint="https://mcp.example.com/mcp",
            )
        ]
    )

    probe_result = await manager.probe_server("demo-http")

    assert probe_result.is_available is True
    assert probe_result.detail == "HTTP MCP 服务可用，共发现 3 个工具"


@pytest.mark.asyncio
async def test_mcp_manager_lists_remote_tools(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器能够读取远端工具并构造对外响应。"""

    async def fake_list_tools(
        self: object,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
    ) -> list[McpClientToolDefinition]:
        """返回稳定的工具列表。"""

        del self, headers
        assert endpoint == "https://mcp.example.com/mcp"
        return [
            McpClientToolDefinition(
                name="weather",
                description="查询天气。",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ]

    monkeypatch.setattr("app.mcp.http_client.McpHttpClient.list_tools", fake_list_tools)
    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="demo-http",
                transport="http",
                endpoint="https://mcp.example.com/mcp",
            )
        ]
    )

    tool_list = await manager.list_tools("demo-http")

    assert tool_list[0].name == "weather"
    assert tool_list[0].registered_name == "mcp_demo_http__weather"


@pytest.mark.asyncio
async def test_mcp_manager_calls_remote_tool(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器能够调用远端工具并返回标准化结果。"""

    async def fake_call_tool(
        self: object,
        *,
        endpoint: str,
        tool_name: str,
        arguments: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> McpClientToolCallResult:
        """返回稳定的工具调用结果。"""

        del self, headers
        assert endpoint == "https://mcp.example.com/mcp"
        assert tool_name == "weather"
        assert arguments == {"city": "杭州"}
        return McpClientToolCallResult(
            content=[{"type": "text", "text": "杭州晴，26 度"}],
            structured_content={"city": "杭州"},
            is_error=False,
            output_text="杭州晴，26 度",
        )

    monkeypatch.setattr("app.mcp.http_client.McpHttpClient.call_tool", fake_call_tool)
    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="demo-http",
                transport="http",
                endpoint="https://mcp.example.com/mcp",
            )
        ]
    )

    tool_result = await manager.call_tool(
        server_name="demo-http",
        tool_name="weather",
        arguments={"city": "杭州"},
    )

    assert tool_result.tool_name == "weather"
    assert tool_result.output_text == "杭州晴，26 度"


@pytest.mark.asyncio
async def test_mcp_manager_calls_sse_tool_with_friendly_error(monkeypatch: MonkeyPatch) -> None:
    """验证 SSE MCP 工具错误会被转换为更直接的诊断信息。"""

    async def fake_call_tool(
        self: object,
        *,
        endpoint: str,
        tool_name: str,
        arguments: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> McpClientToolCallResult:
        """返回稳定的高德 Key 错误。"""

        del self, headers
        assert endpoint == "https://mcp.example.com/sse"
        assert tool_name == "maps_direction_transit_integrated"
        assert arguments == {"origin": "金华", "destination": "杭州"}
        return McpClientToolCallResult(
            content=[
                {
                    "type": "text",
                    "text": "USERKEY_PLAT_NOMATCH",
                }
            ],
            structured_content=None,
            is_error=True,
            output_text="USERKEY_PLAT_NOMATCH",
        )

    monkeypatch.setattr("app.mcp.sse_client.McpSseClient.call_tool", fake_call_tool)
    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="demo-sse",
                transport="sse",
                endpoint="https://mcp.example.com/sse",
            )
        ]
    )

    tool_result = await manager.call_tool(
        server_name="demo-sse",
        tool_name="maps_direction_transit_integrated",
        arguments={"origin": "金华", "destination": "杭州"},
    )

    assert tool_result.is_error is True
    assert "USERKEY_PLAT_NOMATCH" in tool_result.output_text
    assert "高德 Key 与当前调用平台不匹配" in tool_result.output_text


@pytest.mark.asyncio
async def test_mcp_manager_builds_runtime_tools(monkeypatch: MonkeyPatch) -> None:
    """验证 MCP 管理器能够生成供 Agent 直接绑定的运行时工具。"""

    async def fake_list_tools(
        self: object,
        *,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[McpClientToolDefinition]:
        """返回稳定的 stdio 工具列表。"""

        del self, args, cwd, env
        assert command == "uvx"
        return [
            McpClientToolDefinition(
                name="maps_weather",
                description="查询天气。",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    monkeypatch.setattr("app.mcp.stdio_client.McpStdioClient.list_tools", fake_list_tools)
    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="amap",
                transport="stdio",
                command="uvx",
                args=["amap-mcp-server"],
            )
        ]
    )

    runtime_tools = await manager.build_runtime_tools()

    assert runtime_tools[0].registered_name == "mcp_amap__maps_weather"
    assert runtime_tools[0].remote_tool_name == "maps_weather"


def test_mcp_manager_builds_agent_context() -> None:
    """验证 MCP 管理器会生成注入模型的服务说明。"""

    manager = McpManager(
        server_definitions=[
            McpServerDefinition(
                name="demo-http",
                transport="http",
                endpoint="https://mcp.example.com/mcp",
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
