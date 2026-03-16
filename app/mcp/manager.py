"""MCP 管理器模块。

负责管理 MCP 服务器配置、选择不同传输方式客户端并对外暴露统一入口。
当前阶段只提供最小骨架，不负责完整服务注册和会话级连接池。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from json import loads
from typing import Literal

from app.core.config import get_settings
from app.core.exceptions import AppException, ConfigurationException
from app.core.logger import get_logger
from app.mcp.http_client import McpHttpClient
from app.mcp.stdio_client import McpStdioClient
from app.schemas.mcp import McpProbeResponse, McpServerInfo

LOGGER = get_logger(__name__)

McpTransport = Literal["http", "stdio"]


@dataclass(slots=True)
class McpServerDefinition:
    """MCP 服务器配置定义。"""

    name: str
    transport: McpTransport
    endpoint: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    is_enabled: bool = True


class McpManager:
    """MCP 管理器。"""

    def __init__(
        self,
        *,
        server_definitions: list[McpServerDefinition] | None = None,
        http_client: McpHttpClient | None = None,
        stdio_client: McpStdioClient | None = None,
    ) -> None:
        self._settings = get_settings()
        self._http_client = http_client or McpHttpClient()
        self._stdio_client = stdio_client or McpStdioClient()
        self._server_definitions = (
            server_definitions
            if server_definitions is not None
            else self._load_server_definitions()
        )

    def list_servers(self) -> list[McpServerInfo]:
        """返回当前配置的 MCP 服务器列表。"""

        return [
            McpServerInfo(
                name=server_definition.name,
                transport=server_definition.transport,
                endpoint=server_definition.endpoint,
                command=server_definition.command,
                is_enabled=server_definition.is_enabled,
            )
            for server_definition in self._server_definitions
        ]

    async def probe_server(self, server_name: str) -> McpProbeResponse:
        """探测指定 MCP 服务器是否可用。"""

        server_definition = self._get_server_definition(server_name)
        if server_definition.transport == "http":
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "HTTP MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_name},
                )
            is_available, detail = await self._http_client.probe(server_definition.endpoint)
        else:
            if not server_definition.command:
                raise ConfigurationException(
                    "stdio MCP 服务缺少 command 配置。",
                    details={"server_name": server_name},
                )
            is_available, detail = await self._stdio_client.probe(
                command=server_definition.command,
                args=server_definition.args,
                cwd=server_definition.cwd,
            )

        return McpProbeResponse(
            name=server_definition.name,
            transport=server_definition.transport,
            is_available=is_available,
            detail=detail,
        )

    async def call_server_method(
        self,
        *,
        server_name: str,
        method: str,
        params: dict[str, object],
    ) -> object:
        """调用指定 MCP 服务器方法。"""

        server_definition = self._get_server_definition(server_name)
        if server_definition.transport == "http":
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "HTTP MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_name},
                )
            return await self._http_client.call_method(
                endpoint=server_definition.endpoint,
                method=method,
                params=params,
            )

        if not server_definition.command:
            raise ConfigurationException(
                "stdio MCP 服务缺少 command 配置。",
                details={"server_name": server_name},
            )
        return await self._stdio_client.call_method(
            command=server_definition.command,
            args=server_definition.args,
            method=method,
            params=params,
            cwd=server_definition.cwd,
            env=server_definition.env or None,
        )

    def build_agent_context(self) -> str | None:
        """构造用于注入模型的 MCP 服务说明。"""

        available_servers = [server for server in self._server_definitions if server.is_enabled]
        if not available_servers:
            return None

        context_lines = ["以下是当前系统已配置的 MCP 服务骨架信息，可在需要外部能力时参考："]
        for server_definition in available_servers:
            if server_definition.transport == "http":
                context_lines.append(
                    f"- {server_definition.name} [http] endpoint={server_definition.endpoint}"
                )
            else:
                context_lines.append(
                    f"- {server_definition.name} [stdio] command={server_definition.command}"
                )
        context_lines.append("当前阶段仅完成 MCP 骨架与管理接口，尚未自动执行远端 MCP tool。")
        return "\n".join(context_lines)

    def _get_server_definition(self, server_name: str) -> McpServerDefinition:
        """按名称获取 MCP 服务器配置。"""

        for server_definition in self._server_definitions:
            if server_definition.name == server_name:
                return server_definition
        raise AppException(
            "指定的 MCP 服务器不存在。",
            error_code="mcp_server_not_found",
            details={"server_name": server_name},
        )

    def _load_server_definitions(self) -> list[McpServerDefinition]:
        """从环境变量加载 MCP 服务器配置。"""

        raw_mcp_servers_json = self._settings.mcp_servers_json
        if raw_mcp_servers_json is None or not raw_mcp_servers_json.strip():
            return []

        try:
            parsed_payload = loads(raw_mcp_servers_json)
        except ValueError as exception:
            raise ConfigurationException(
                "MCP_SERVERS_JSON 不是合法 JSON。",
                details={"config_key": "MCP_SERVERS_JSON"},
            ) from exception

        if not isinstance(parsed_payload, list):
            raise ConfigurationException(
                "MCP_SERVERS_JSON 必须是 JSON 数组。",
                details={"config_key": "MCP_SERVERS_JSON"},
            )

        server_definitions: list[McpServerDefinition] = []
        for server_payload in parsed_payload:
            if not isinstance(server_payload, dict):
                LOGGER.warning("检测到非法 MCP 配置项，已忽略。", extra={"payload": server_payload})
                continue
            transport = str(server_payload.get("transport", "http"))
            if transport not in {"http", "stdio"}:
                LOGGER.warning(
                    "检测到未知 MCP 传输方式，已忽略。", extra={"payload": server_payload}
                )
                continue
            args_payload = server_payload.get("args", [])
            env_payload = server_payload.get("env", {})
            server_definitions.append(
                McpServerDefinition(
                    name=str(server_payload.get("name", "")),
                    transport=transport,
                    endpoint=(
                        str(server_payload["endpoint"])
                        if server_payload.get("endpoint") is not None
                        else None
                    ),
                    command=(
                        str(server_payload["command"])
                        if server_payload.get("command") is not None
                        else None
                    ),
                    args=[str(argument) for argument in args_payload]
                    if isinstance(args_payload, list)
                    else [],
                    cwd=str(server_payload["cwd"])
                    if server_payload.get("cwd") is not None
                    else None,
                    env=(
                        {str(key): str(value) for key, value in env_payload.items()}
                        if isinstance(env_payload, dict)
                        else {}
                    ),
                    is_enabled=bool(server_payload.get("enabled", True)),
                )
            )
        return server_definitions
