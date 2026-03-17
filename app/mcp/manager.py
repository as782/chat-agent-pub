"""MCP 管理器模块。
负责管理 MCP 服务配置、选择传输方式客户端并对外暴露统一入口。
当前阶段重点是最小可用能力：标准握手、工具发现、工具调用与 Agent 工具桥接。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from json import dumps, loads
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import AppException, ConfigurationException
from app.core.logger import get_logger
from app.mcp.http_client import McpHttpClient
from app.mcp.models import (
    McpClientToolCallResult,
    McpClientToolDefinition,
    McpRuntimeTool,
    McpTransport,
)
from app.mcp.sse_client import McpSseClient
from app.mcp.stdio_client import McpStdioClient
from app.schemas.mcp import McpProbeResponse, McpServerInfo, McpToolCallResponse, McpToolInfo

LOGGER = get_logger(__name__)

MCP_TOOL_NAME_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


@dataclass(slots=True)
class McpServerDefinition:
    """MCP 服务配置定义。"""

    name: str
    transport: McpTransport
    endpoint: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    is_enabled: bool = True


class McpManager:
    """MCP 管理器。"""

    def __init__(
        self,
        *,
        server_definitions: list[McpServerDefinition] | None = None,
        http_client: McpHttpClient | None = None,
        sse_client: McpSseClient | None = None,
        stdio_client: McpStdioClient | None = None,
    ) -> None:
        self._settings = get_settings()
        self._http_client = http_client or McpHttpClient()
        self._sse_client = sse_client or McpSseClient()
        self._stdio_client = stdio_client or McpStdioClient()
        self._server_definitions = (
            server_definitions
            if server_definitions is not None
            else self._load_server_definitions()
        )

    def list_servers(self) -> list[McpServerInfo]:
        """返回当前配置的 MCP 服务列表。"""

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
        """探测指定 MCP 服务是否可用。"""

        server_definition = self._get_server_definition(server_name)
        if server_definition.transport in {"http", "streamable_http"}:
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "HTTP MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_name},
                )
            is_available, detail = await self._http_client.probe(
                server_definition.endpoint,
                headers=server_definition.headers or None,
            )
        elif server_definition.transport == "sse":
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "SSE MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_name},
                )
            is_available, detail = await self._sse_client.probe(
                server_definition.endpoint,
                headers=server_definition.headers or None,
            )
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
                env=server_definition.env or None,
            )

        return McpProbeResponse(
            name=server_definition.name,
            transport=server_definition.transport,
            is_available=is_available,
            detail=detail,
        )

    async def list_tools(self, server_name: str) -> list[McpToolInfo]:
        """列出指定 MCP 服务提供的工具。"""

        server_definition = self._get_server_definition(server_name)
        tool_definitions = await self._list_remote_tools(server_definition)
        return [
            McpToolInfo(
                server_name=server_name,
                name=tool_definition.name,
                description=tool_definition.description,
                input_schema=tool_definition.input_schema,
                registered_name=self._build_registered_tool_name(
                    server_name=server_name,
                    remote_tool_name=tool_definition.name,
                ),
            )
            for tool_definition in tool_definitions
        ]

    async def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> McpToolCallResponse:
        """调用指定 MCP 服务的工具。"""

        server_definition = self._get_server_definition(server_name)
        call_result = await self._call_remote_tool(
            server_definition=server_definition,
            tool_name=tool_name,
            arguments=arguments,
        )
        normalized_output_text = self._build_tool_output_text(call_result)
        return McpToolCallResponse(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
            content=call_result.content,
            structured_content=call_result.structured_content,
            is_error=call_result.is_error,
            output_text=normalized_output_text,
        )

    async def call_server_method(
        self,
        *,
        server_name: str,
        method: str,
        params: dict[str, object],
    ) -> object:
        """兼容旧接口的通用方法调用。"""

        if method == "tools/list":
            return {
                "tools": [
                    tool_info.model_dump(mode="json")
                    for tool_info in await self.list_tools(server_name)
                ]
            }

        if method == "tools/call":
            tool_name = params.get("name")
            tool_arguments = params.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise AppException(
                    "调用 tools/call 时必须传入 name。",
                    error_code="invalid_request",
                )
            if not isinstance(tool_arguments, dict):
                raise AppException(
                    "调用 tools/call 时 arguments 必须是 JSON 对象。",
                    error_code="invalid_request",
                )
            return (
                await self.call_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=tool_arguments,
                )
            ).model_dump(mode="json")

        raise AppException(
            "当前通用 MCP 调用仅支持 tools/list 和 tools/call。",
            error_code="unsupported_mcp_method",
            details={"method": method},
        )

    async def build_runtime_tools(
        self,
        *,
        server_names: list[str] | None = None,
    ) -> list[McpRuntimeTool]:
        """构造供 Agent 直接绑定给模型的 MCP 工具列表。"""

        selected_server_definitions = self._select_server_definitions(server_names)
        runtime_tools: list[McpRuntimeTool] = []
        seen_registered_names: set[str] = set()

        for server_definition in selected_server_definitions:
            tool_definitions = await self._list_remote_tools(server_definition)
            for tool_definition in tool_definitions:
                registered_name = self._build_registered_tool_name(
                    server_name=server_definition.name,
                    remote_tool_name=tool_definition.name,
                )
                if registered_name in seen_registered_names:
                    raise AppException(
                        "不同 MCP 服务产生了重复的工具注册名。",
                        error_code="mcp_tool_name_conflict",
                        details={"registered_name": registered_name},
                    )

                seen_registered_names.add(registered_name)
                runtime_tools.append(
                    McpRuntimeTool(
                        registered_name=registered_name,
                        server_name=server_definition.name,
                        remote_tool_name=tool_definition.name,
                        description=tool_definition.description,
                        input_schema=tool_definition.input_schema,
                    )
                )

        return runtime_tools

    def build_agent_context(self, runtime_tools: list[McpRuntimeTool] | None = None) -> str | None:
        """构造注入模型的 MCP 上下文说明。"""

        available_servers = [server for server in self._server_definitions if server.is_enabled]
        if not available_servers:
            return None

        context_lines = [
            "以下是当前系统已接入的 MCP 服务与工具信息，必要时可以优先选择合适的 MCP 工具完成查询："
        ]
        for server_definition in available_servers:
            if server_definition.transport in {"http", "streamable_http"}:
                context_lines.append(
                    f"- 服务 {server_definition.name} [http] endpoint={server_definition.endpoint}"
                )
            elif server_definition.transport == "sse":
                context_lines.append(
                    f"- 服务 {server_definition.name} [sse] endpoint={server_definition.endpoint}"
                )
            else:
                context_lines.append(
                    f"- 服务 {server_definition.name} [stdio] command={server_definition.command}"
                )

        if runtime_tools:
            context_lines.append("可直接调用的 MCP 工具：")
            for runtime_tool in runtime_tools:
                context_lines.append(
                    f"- {runtime_tool.registered_name}: {runtime_tool.description or '无描述'}"
                )

        return "\n".join(context_lines)

    def find_runtime_tool(
        self,
        *,
        runtime_tools: list[McpRuntimeTool],
        registered_name: str,
    ) -> McpRuntimeTool | None:
        """按注册名查找运行时工具。"""

        for runtime_tool in runtime_tools:
            if runtime_tool.registered_name == registered_name:
                return runtime_tool
        return None

    def _get_server_definition(self, server_name: str) -> McpServerDefinition:
        """按名称获取 MCP 服务配置。"""

        for server_definition in self._server_definitions:
            if server_definition.name == server_name:
                return server_definition

        raise AppException(
            "指定的 MCP 服务不存在。",
            error_code="mcp_server_not_found",
            details={"server_name": server_name},
        )

    def _select_server_definitions(
        self,
        server_names: list[str] | None,
    ) -> list[McpServerDefinition]:
        """选择需要参与本次执行的 MCP 服务列表。"""

        if server_names:
            return [self._get_server_definition(server_name) for server_name in server_names]
        return [server for server in self._server_definitions if server.is_enabled]

    async def _list_remote_tools(
        self,
        server_definition: McpServerDefinition,
    ) -> list[McpClientToolDefinition]:
        """按服务配置列出远端工具。"""

        if server_definition.transport in {"http", "streamable_http"}:
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "HTTP MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_definition.name},
                )
            return await self._http_client.list_tools(
                endpoint=server_definition.endpoint,
                headers=server_definition.headers or None,
            )
        if server_definition.transport == "sse":
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "SSE MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_definition.name},
                )
            return await self._sse_client.list_tools(
                endpoint=server_definition.endpoint,
                headers=server_definition.headers or None,
            )

        if not server_definition.command:
            raise ConfigurationException(
                "stdio MCP 服务缺少 command 配置。",
                details={"server_name": server_definition.name},
            )
        return await self._stdio_client.list_tools(
            command=server_definition.command,
            args=server_definition.args,
            cwd=server_definition.cwd,
            env=server_definition.env or None,
        )

    async def _call_remote_tool(
        self,
        *,
        server_definition: McpServerDefinition,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> McpClientToolCallResult:
        """按服务配置调用远端工具。"""

        if server_definition.transport in {"http", "streamable_http"}:
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "HTTP MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_definition.name},
                )
            return await self._http_client.call_tool(
                endpoint=server_definition.endpoint,
                tool_name=tool_name,
                arguments=arguments,
                headers=server_definition.headers or None,
            )
        if server_definition.transport == "sse":
            if not server_definition.endpoint:
                raise ConfigurationException(
                    "SSE MCP 服务缺少 endpoint 配置。",
                    details={"server_name": server_definition.name},
                )
            return await self._sse_client.call_tool(
                endpoint=server_definition.endpoint,
                tool_name=tool_name,
                arguments=arguments,
                headers=server_definition.headers or None,
            )

        if not server_definition.command:
            raise ConfigurationException(
                "stdio MCP 服务缺少 command 配置。",
                details={"server_name": server_definition.name},
            )
        return await self._stdio_client.call_tool(
            command=server_definition.command,
            args=server_definition.args,
            tool_name=tool_name,
            arguments=arguments,
            cwd=server_definition.cwd,
            env=server_definition.env or None,
        )

    @staticmethod
    def _build_registered_tool_name(*, server_name: str, remote_tool_name: str) -> str:
        """构造可安全绑定给模型的 MCP 工具名。"""

        normalized_server_name = MCP_TOOL_NAME_SANITIZE_PATTERN.sub("_", server_name).strip("_")
        normalized_remote_tool_name = MCP_TOOL_NAME_SANITIZE_PATTERN.sub(
            "_",
            remote_tool_name,
        ).strip("_")
        normalized_server_name = normalized_server_name or "server"
        normalized_remote_tool_name = normalized_remote_tool_name or "tool"
        return f"mcp_{normalized_server_name}__{normalized_remote_tool_name}"

    def _load_server_definitions(self) -> list[McpServerDefinition]:
        """从环境变量加载 MCP 服务配置。"""

        raw_mcp_servers_json = self._settings.mcp_servers_json
        if raw_mcp_servers_json is None or not raw_mcp_servers_json.strip():
            return []

        try:
            parsed_payload = loads(raw_mcp_servers_json)
        except ValueError as exception:
            raise ConfigurationException(
                "MCP_SERVERS_JSON 不是合法 JSON。"
                "请在 .env 中使用单行 JSON，或使用带引号的完整 JSON 字符串。",
                details={
                    "config_key": "MCP_SERVERS_JSON",
                    "example": '{"mcpServers":{"amap-maps-streamableHTTP":{"url":"https://mcp.amap.com/mcp?key=replace-me"}}}',
                },
            ) from exception

        flat_server_payloads = self._flatten_server_payloads(parsed_payload)

        server_definitions: list[McpServerDefinition] = []
        for server_payload in flat_server_payloads:
            if not isinstance(server_payload, dict):
                LOGGER.warning("检测到非法 MCP 配置项，已忽略。", extra={"payload": server_payload})
                continue

            raw_endpoint = server_payload.get("endpoint")
            if raw_endpoint is None:
                raw_endpoint = server_payload.get("url")

            transport = self._resolve_transport(server_payload, raw_endpoint)
            if transport is None:
                LOGGER.warning(
                    "检测到未知 MCP 传输方式，已忽略。", extra={"payload": server_payload}
                )
                continue

            server_name = str(server_payload.get("name", "")).strip()
            if not server_name:
                LOGGER.warning(
                    "检测到缺少名称的 MCP 配置项，已忽略。", extra={"payload": server_payload}
                )
                continue

            args_payload = server_payload.get("args", [])
            env_payload = server_payload.get("env", {})
            headers_payload = server_payload.get("headers", {})
            server_definitions.append(
                McpServerDefinition(
                    name=server_name,
                    transport=transport,
                    endpoint=str(raw_endpoint) if raw_endpoint is not None else None,
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
                    headers=(
                        {str(key): str(value) for key, value in headers_payload.items()}
                        if isinstance(headers_payload, dict)
                        else {}
                    ),
                    is_enabled=bool(server_payload.get("enabled", True)),
                )
            )
        return server_definitions

    def _flatten_server_payloads(self, parsed_payload: object) -> list[dict[str, object]]:
        """兼容多种 MCP 配置 JSON 结构并展开为统一的扁平列表。"""

        if isinstance(parsed_payload, list):
            flat_payloads: list[dict[str, object]] = []
            for item in parsed_payload:
                if isinstance(item, dict) and "mcpServers" in item:
                    flat_payloads.extend(self._expand_mcp_servers_mapping(item["mcpServers"]))
                    continue

                if isinstance(item, dict):
                    flat_payloads.append(item)
                    continue

                LOGGER.warning("检测到非法 MCP 配置项，已忽略。", extra={"payload": item})
            return flat_payloads

        if isinstance(parsed_payload, dict) and "mcpServers" in parsed_payload:
            return self._expand_mcp_servers_mapping(parsed_payload["mcpServers"])

        raise ConfigurationException(
            "MCP_SERVERS_JSON 必须是 JSON 数组，或包含 mcpServers 的 JSON 对象。",
            details={"config_key": "MCP_SERVERS_JSON"},
        )

    def _expand_mcp_servers_mapping(self, mcp_servers_payload: object) -> list[dict[str, object]]:
        """展开高德文档常见的 mcpServers 映射格式。"""

        if not isinstance(mcp_servers_payload, dict):
            raise ConfigurationException(
                "mcpServers 必须是 JSON 对象。",
                details={"config_key": "MCP_SERVERS_JSON", "field": "mcpServers"},
            )

        flat_payloads: list[dict[str, object]] = []
        for server_name, server_config in mcp_servers_payload.items():
            if not isinstance(server_config, dict):
                LOGGER.warning(
                    "检测到非法 MCP 服务定义，已忽略。",
                    extra={"server_name": str(server_name), "payload": server_config},
                )
                continue

            flat_payloads.append(
                {
                    "name": str(server_name),
                    "transport": server_config.get("transport"),
                    "endpoint": server_config.get("endpoint"),
                    "url": server_config.get("url"),
                    "command": server_config.get("command"),
                    "args": server_config.get("args", []),
                    "cwd": server_config.get("cwd"),
                    "env": server_config.get("env", {}),
                    "headers": server_config.get("headers", {}),
                    "enabled": server_config.get("enabled", True),
                }
            )

        return flat_payloads

    def _resolve_transport(
        self,
        server_payload: dict[str, object],
        raw_endpoint: object,
    ) -> McpTransport | None:
        """根据配置内容推断 MCP 传输方式。"""

        normalized_transport = self._normalize_transport(
            str(server_payload.get("transport", "")).strip().lower()
        )
        if normalized_transport is not None:
            return normalized_transport

        if server_payload.get("command") is not None:
            return "stdio"
        if raw_endpoint is not None:
            server_name = str(server_payload.get("name", "")).lower()
            endpoint_text = str(raw_endpoint).lower()
            if "sse" in server_name or "/sse" in endpoint_text:
                return "sse"
            return "http"
        return None

    @staticmethod
    def _normalize_transport(raw_transport: str) -> McpTransport | None:
        """统一解析传输方式别名。"""

        if raw_transport == "http":
            return "http"
        if raw_transport in {"streamable_http", "streamable-http", "streamablehttp"}:
            return "streamable_http"
        if raw_transport == "sse":
            return "sse"
        if raw_transport == "stdio":
            return "stdio"
        return None

    @staticmethod
    def _build_tool_output_text(call_result: McpClientToolCallResult) -> str:
        """构造适合返回给 API 和模型的工具输出文本。"""

        raw_output_text = call_result.output_text.strip()
        if not raw_output_text and call_result.structured_content is not None:
            raw_output_text = dumps(call_result.structured_content, ensure_ascii=False)

        if not call_result.is_error:
            return raw_output_text

        if "USERKEY_PLAT_NOMATCH" in raw_output_text.upper():
            return (
                "MCP 工具调用失败：高德 Key 与当前调用平台不匹配"
                "（USERKEY_PLAT_NOMATCH）。请在高德开放平台检查该 Key 的应用类型，"
                "通常需要使用与 Web 服务或 MCP 远程调用兼容的 Key。"
                f" 原始错误：{raw_output_text}"
            )

        if raw_output_text:
            return f"MCP 工具调用失败：{raw_output_text}"
        return "MCP 工具调用失败，远端服务未返回更多信息。"
