"""MCP stdio 客户端模块。
负责通过官方 MCP Python SDK 与 stdio 类型服务建立标准会话。
当前阶段不负责长连接池化，只保证单次调用链路的标准握手与工具执行。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from shutil import which
from typing import Any

from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.core.exceptions import UpstreamServiceException
from app.mcp.models import McpClientToolCallResult, McpClientToolDefinition

DEFAULT_MCP_STDIO_TIMEOUT_SECONDS = 10.0


class McpStdioClient:
    """基于官方 SDK 的 MCP stdio 客户端。"""

    async def probe(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """通过标准 initialize 与 tools/list 探测 stdio MCP 服务。"""

        if which(command) is None and not Path(command).exists():
            return False, f"命令 {command} 不存在或不可执行"

        try:
            tools = await self.list_tools(
                command=command,
                args=args,
                cwd=cwd,
                env=env,
            )
        except UpstreamServiceException as exception:
            return False, exception.message

        return True, f"stdio MCP 服务可用，共发现 {len(tools)} 个工具"

    async def list_tools(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[McpClientToolDefinition]:
        """列出 stdio MCP 服务暴露的工具。"""

        async with self._open_session(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
        ) as session:
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
        command: str,
        args: list[str],
        tool_name: str,
        arguments: dict[str, Any],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> McpClientToolCallResult:
        """调用 stdio MCP 服务中的指定工具。"""

        async with self._open_session(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
        ) as session:
            call_result = await session.call_tool(tool_name, arguments=arguments)

        return self._normalize_call_result(call_result)

    @asynccontextmanager
    async def _open_session(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        """打开一次标准 MCP stdio 会话。"""

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        server_parameters = StdioServerParameters(
            command=command,
            args=args,
            cwd=cwd,
            env=merged_env,
        )

        try:
            async with stdio_client(server_parameters) as (read_stream, write_stream):
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
                "stdio MCP 服务初始化失败。",
                error_code="mcp_stdio_session_error",
                details={"command": command},
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
