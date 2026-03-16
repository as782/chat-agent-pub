"""MCP HTTP 客户端模块。
负责通过官方 MCP Python SDK 与 streamable-http 类型服务建立标准会话。
当前阶段保留 `http` 作为 `streamable_http` 的兼容别名。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

from app.core.exceptions import UpstreamServiceException
from app.mcp.models import McpClientToolCallResult, McpClientToolDefinition

DEFAULT_MCP_HTTP_TIMEOUT_SECONDS = 10.0


class McpHttpClient:
    """基于官方 SDK 的 streamable-http MCP 客户端。"""

    async def probe(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """通过标准 initialize 与 tools/list 探测 HTTP MCP 服务。"""

        try:
            tools = await self.list_tools(endpoint=endpoint, headers=headers)
        except UpstreamServiceException as exception:
            return False, exception.message

        return True, f"HTTP MCP 服务可用，共发现 {len(tools)} 个工具"

    async def list_tools(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
    ) -> list[McpClientToolDefinition]:
        """列出 HTTP MCP 服务暴露的工具。"""

        async with self._open_session(endpoint=endpoint, headers=headers) as session:
            list_result = await session.list_tools()

        return [
            McpClientToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
                output_schema=tool.outputSchema,
            )
            for tool in list_result.tools
        ]

    async def call_tool(
        self,
        *,
        endpoint: str,
        tool_name: str,
        arguments: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> McpClientToolCallResult:
        """调用 HTTP MCP 服务中的指定工具。"""

        async with self._open_session(endpoint=endpoint, headers=headers) as session:
            call_result = await session.call_tool(tool_name, arguments=arguments)

        return self._normalize_call_result(call_result)

    @asynccontextmanager
    async def _open_session(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
    ):
        """打开一次标准 MCP HTTP 会话。"""

        try:
            async with streamablehttp_client(
                endpoint,
                headers=headers,
                timeout=DEFAULT_MCP_HTTP_TIMEOUT_SECONDS,
            ) as (read_stream, write_stream, _session_id_callback):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    client_info=types.Implementation(
                        name="chat-agent-backend",
                        version="0.1.0",
                    ),
                ) as session:
                    await session.initialize()
                    yield session
        except UpstreamServiceException:
            raise
        except Exception as exception:
            raise UpstreamServiceException(
                "HTTP MCP 服务初始化失败。",
                error_code="mcp_http_session_error",
                details={"endpoint": endpoint},
            ) from exception

    @staticmethod
    def _normalize_call_result(call_result: types.CallToolResult) -> McpClientToolCallResult:
        """标准化 MCP tool 调用结果，便于 API 和 Agent 共用。"""

        serialized_content = [
            content_item.model_dump(mode="json") for content_item in call_result.content
        ]
        text_fragments: list[str] = []
        for content_item in call_result.content:
            if isinstance(content_item, types.TextContent) and content_item.text:
                text_fragments.append(content_item.text)
            else:
                text_fragments.append(content_item.model_dump_json())

        return McpClientToolCallResult(
            content=serialized_content,
            structured_content=call_result.structuredContent,
            is_error=call_result.isError,
            output_text="\n".join(fragment for fragment in text_fragments if fragment).strip(),
        )
