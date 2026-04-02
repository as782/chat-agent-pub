"""HTTP transport client for MCP streamable-http services."""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

from app.core.config import get_settings
from app.core.exceptions import UpstreamServiceException
from app.core.logger import get_logger
from app.mcp.models import McpClientToolCallResult, McpClientToolDefinition

LOGGER = get_logger(__name__)


class McpHttpClient:
    """Client for MCP streamable-http services."""

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

        return True, f"HTTP MCP service is available and exposes {len(tools)} tools."

    async def list_tools(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
    ) -> list[McpClientToolDefinition]:
        request_start_time = perf_counter()
        timeout_seconds = self._settings.mcp_http_timeout_seconds
        LOGGER.info(
            "MCP HTTP list_tools started: endpoint=%s connect_timeout_seconds=%.2f",
            endpoint,
            timeout_seconds,
        )

        try:
            async with self._open_session(endpoint=endpoint, headers=headers) as session:
                list_result = await session.list_tools()
        except Exception:
            LOGGER.warning(
                "MCP HTTP list_tools failed: endpoint=%s duration_ms=%.2f connect_timeout_seconds=%.2f",
                endpoint,
                (perf_counter() - request_start_time) * 1000,
                timeout_seconds,
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
            "MCP HTTP list_tools completed: endpoint=%s duration_ms=%.2f connect_timeout_seconds=%.2f tool_count=%s",
            endpoint,
            (perf_counter() - request_start_time) * 1000,
            timeout_seconds,
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
        timeout_seconds = self._settings.mcp_http_timeout_seconds
        LOGGER.info(
            "MCP HTTP call_tool started: endpoint=%s tool=%s connect_timeout_seconds=%.2f",
            endpoint,
            tool_name,
            timeout_seconds,
        )

        try:
            async with self._open_session(endpoint=endpoint, headers=headers) as session:
                call_result = await session.call_tool(tool_name, arguments=arguments)
        except Exception:
            LOGGER.warning(
                (
                    "MCP HTTP call_tool failed: endpoint=%s tool=%s duration_ms=%.2f "
                    "connect_timeout_seconds=%.2f"
                ),
                endpoint,
                tool_name,
                (perf_counter() - request_start_time) * 1000,
                timeout_seconds,
            )
            raise

        normalized_result = self._normalize_call_result(call_result)
        LOGGER.info(
            (
                "MCP HTTP call_tool completed: endpoint=%s tool=%s duration_ms=%.2f "
                "connect_timeout_seconds=%.2f is_error=%s"
            ),
            endpoint,
            tool_name,
            (perf_counter() - request_start_time) * 1000,
            timeout_seconds,
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
        timeout_seconds = self._settings.mcp_http_timeout_seconds
        try:
            async with streamablehttp_client(
                endpoint,
                headers=headers,
                timeout=timeout_seconds,
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
            LOGGER.warning(
                "MCP HTTP session init failed: endpoint=%s connect_timeout_seconds=%.2f error_type=%s",
                endpoint,
                timeout_seconds,
                type(exception).__name__,
            )
            raise UpstreamServiceException(
                "HTTP MCP session initialization failed.",
                error_code="mcp_http_session_error",
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
