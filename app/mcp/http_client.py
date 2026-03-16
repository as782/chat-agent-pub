"""MCP HTTP 客户端模块。

负责通过 HTTP 方式探测和调用 MCP 服务器。
当前阶段按最小 JSON-RPC 骨架实现，不负责完整协议协商。
"""

from __future__ import annotations

from uuid import uuid4

import httpx

from app.core.exceptions import UpstreamServiceException

DEFAULT_MCP_HTTP_TIMEOUT_SECONDS = 5.0


class McpHttpClient:
    """MCP HTTP 客户端。"""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._http_client = http_client

    async def probe(self, endpoint: str) -> tuple[bool, str]:
        """探测 HTTP MCP 服务当前是否可达。"""

        try:
            if self._http_client is not None:
                response = await self._http_client.get(endpoint)
            else:
                async with httpx.AsyncClient(
                    timeout=DEFAULT_MCP_HTTP_TIMEOUT_SECONDS
                ) as http_client:
                    response = await http_client.get(endpoint)
        except httpx.HTTPError as exception:
            return False, f"HTTP 连接失败：{exception.__class__.__name__}"

        if response.status_code < 500:
            return True, f"HTTP 服务可达，状态码 {response.status_code}"
        return False, f"HTTP 服务返回异常状态码 {response.status_code}"

    async def call_method(
        self,
        *,
        endpoint: str,
        method: str,
        params: dict[str, object],
    ) -> object:
        """通过最小 JSON-RPC 形式调用 HTTP MCP 服务。"""

        request_payload = {
            "jsonrpc": "2.0",
            "id": uuid4().hex,
            "method": method,
            "params": params,
        }
        try:
            if self._http_client is not None:
                response = await self._http_client.post(endpoint, json=request_payload)
            else:
                async with httpx.AsyncClient(
                    timeout=DEFAULT_MCP_HTTP_TIMEOUT_SECONDS
                ) as http_client:
                    response = await http_client.post(endpoint, json=request_payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exception:
            raise UpstreamServiceException(
                "HTTP MCP 服务返回了非成功状态码。",
                error_code="mcp_http_error",
                status_code=exception.response.status_code,
                details={"endpoint": endpoint, "response_text": exception.response.text},
            ) from exception
        except httpx.HTTPError as exception:
            raise UpstreamServiceException(
                "调用 HTTP MCP 服务失败。",
                error_code="mcp_http_connection_error",
                details={"endpoint": endpoint},
            ) from exception

        response_payload = response.json()
        if isinstance(response_payload, dict) and "error" in response_payload:
            raise UpstreamServiceException(
                "HTTP MCP 服务返回了业务错误。",
                error_code="mcp_http_business_error",
                details={"endpoint": endpoint, "response": response_payload},
            )
        if isinstance(response_payload, dict) and "result" in response_payload:
            return response_payload["result"]
        return response_payload
