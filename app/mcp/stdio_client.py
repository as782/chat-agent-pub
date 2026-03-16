"""MCP stdio 客户端模块。

负责通过 stdio 方式探测和调用本地 MCP 服务器。
当前阶段按最小 JSON-RPC 骨架实现，不负责长连接复用与高级握手。
"""

from __future__ import annotations

import asyncio
from json import dumps, loads
from pathlib import Path
from shutil import which
from uuid import uuid4

from app.core.exceptions import UpstreamServiceException

DEFAULT_MCP_STDIO_TIMEOUT_SECONDS = 5.0


class McpStdioClient:
    """MCP stdio 客户端。"""

    async def probe(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str | None = None,
    ) -> tuple[bool, str]:
        """探测 stdio MCP 服务器命令是否可执行。"""

        if which(command) or Path(command).exists():
            return True, f"命令 {command} 可执行"
        return False, f"命令 {command} 不存在"

    async def call_method(
        self,
        *,
        command: str,
        args: list[str],
        method: str,
        params: dict[str, object],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> object:
        """通过最小 JSON-RPC 形式调用 stdio MCP 服务。"""

        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if process.stdin is None or process.stdout is None:
                raise UpstreamServiceException(
                    "stdio MCP 服务未正确暴露标准输入输出。",
                    error_code="mcp_stdio_error",
                )

            request_payload = {
                "jsonrpc": "2.0",
                "id": uuid4().hex,
                "method": method,
                "params": params,
            }
            process.stdin.write((dumps(request_payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await process.stdin.drain()
            raw_response = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=DEFAULT_MCP_STDIO_TIMEOUT_SECONDS,
            )
        except TimeoutError as exception:
            raise UpstreamServiceException(
                "stdio MCP 服务响应超时。",
                error_code="mcp_stdio_timeout",
                details={"command": command},
            ) from exception
        except OSError as exception:
            raise UpstreamServiceException(
                "启动 stdio MCP 服务失败。",
                error_code="mcp_stdio_spawn_error",
                details={"command": command},
            ) from exception
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()

        if not raw_response:
            raise UpstreamServiceException(
                "stdio MCP 服务未返回任何输出。",
                error_code="mcp_stdio_empty_response",
                details={"command": command},
            )

        try:
            response_payload = loads(raw_response.decode("utf-8"))
        except ValueError as exception:
            raise UpstreamServiceException(
                "stdio MCP 服务返回了无法解析的 JSON。",
                error_code="mcp_stdio_invalid_response",
                details={"command": command},
            ) from exception

        if isinstance(response_payload, dict) and "error" in response_payload:
            raise UpstreamServiceException(
                "stdio MCP 服务返回了业务错误。",
                error_code="mcp_stdio_business_error",
                details={"command": command, "response": response_payload},
            )
        if isinstance(response_payload, dict) and "result" in response_payload:
            return response_payload["result"]
        return response_payload
