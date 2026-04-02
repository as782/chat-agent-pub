"""SSE transport client for MCP services."""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from mcp import ClientSession, types
from mcp.client.sse import sse_client

from app.core.config import get_settings
from app.core.exceptions import UpstreamServiceException
from app.core.logger import get_logger
from app.mcp.models import McpClientToolCallResult, McpClientToolDefinition

LOGGER = get_logger(__name__)


class McpSseClient:
    """Client for MCP SSE services."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def probe(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        try:
            tools = await self.list_tools(endpoint=endpoint, headers=headers)
        except UpstreamServiceException as exception:
            return False, exception.message

        return True, f"SSE MCP service is available and exposes {len(tools)} tools."

    async def list_tools(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
    ) -> list[McpClientToolDefinition]:
        request_start_time = perf_counter()
        connect_timeout = self._settings.mcp_sse_timeout_seconds
        read_timeout = self._settings.mcp_sse_read_timeout_seconds
        LOGGER.info(
            (
                "MCP SSE list_tools started: endpoint=%s connect_timeout_seconds=%.2f "
                "read_timeout_seconds=%.2f"
            ),
            endpoint,
            connect_timeout,
            read_timeout,
        )

        try:
            async with self._open_session(endpoint=endpoint, headers=headers) as session:
                list_result = await session.list_tools()
        except Exception:
            LOGGER.warning(
                (
                    "MCP SSE list_tools failed: endpoint=%s duration_ms=%.2f connect_timeout_seconds=%.2f "
                    "read_timeout_seconds=%.2f"
                ),
                endpoint,
                (perf_counter() - request_start_time) * 1000,
                connect_timeout,
                read_timeout,
            )
            raise

        tool_definitions = [
            McpClientToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
                output_schema=tool.outputSchema,
            )
            for tool in list_result.tools
        ]
        LOGGER.info(
            (
                    "MCP SSE list_tools completed: endpoint=%s duration_ms=%.2f connect_timeout_seconds=%.2f "
                "read_timeout_seconds=%.2f tool_count=%s"
            ),
            endpoint,
            (perf_counter() - request_start_time) * 1000,
            connect_timeout,
            read_timeout,
            len(tool_definitions),
        )
        return tool_definitions

    async def call_tool(
        self,
        *,
        endpoint: str,
        tool_name: str,
        arguments: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> McpClientToolCallResult:
        request_start_time = perf_counter()
        connect_timeout = self._settings.mcp_sse_timeout_seconds
        read_timeout = self._settings.mcp_sse_read_timeout_seconds
        LOGGER.info(
            (
                "MCP SSE call_tool started: endpoint=%s tool=%s connect_timeout_seconds=%.2f "
                "read_timeout_seconds=%.2f"
            ),
            endpoint,
            tool_name,
            connect_timeout,
            read_timeout,
        )

        try:
            async with self._open_session(endpoint=endpoint, headers=headers) as session:
                call_result = await session.call_tool(tool_name, arguments=arguments)
        except Exception:
            LOGGER.warning(
                (
                    "MCP SSE call_tool failed: endpoint=%s tool=%s duration_ms=%.2f "
                    "connect_timeout_seconds=%.2f read_timeout_seconds=%.2f"
                ),
                endpoint,
                tool_name,
                (perf_counter() - request_start_time) * 1000,
                connect_timeout,
                read_timeout,
            )
            raise

        normalized_result = self._normalize_call_result(call_result)
        LOGGER.info(
            (
                "MCP SSE call_tool completed: endpoint=%s tool=%s duration_ms=%.2f "
                "connect_timeout_seconds=%.2f read_timeout_seconds=%.2f is_error=%s"
            ),
            endpoint,
            tool_name,
            (perf_counter() - request_start_time) * 1000,
            connect_timeout,
            read_timeout,
            normalized_result.is_error,
        )
        return normalized_result

    @asynccontextmanager
    async def _open_session(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
    ):
        connect_timeout = self._settings.mcp_sse_timeout_seconds
        read_timeout = self._settings.mcp_sse_read_timeout_seconds
        try:
            async with sse_client(
                endpoint,
                headers=headers,
                timeout=connect_timeout,
                sse_read_timeout=read_timeout,
            ) as (read_stream, write_stream):
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
            LOGGER.warning(
                (
                    "MCP SSE session init failed: endpoint=%s connect_timeout_seconds=%.2f "
                    "read_timeout_seconds=%.2f error_type=%s"
                ),
                endpoint,
                connect_timeout,
                read_timeout,
                type(exception).__name__,
            )
            raise UpstreamServiceException(
                "SSE MCP session initialization failed.",
                error_code="mcp_sse_session_error",
                details={"endpoint": endpoint},
            ) from exception

    @staticmethod
    def _normalize_call_result(call_result: types.CallToolResult) -> McpClientToolCallResult:
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
