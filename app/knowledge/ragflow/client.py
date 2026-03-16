"""RAGFlow 基础客户端模块。

负责统一封装对 RAGFlow HTTP API 的认证、请求发送和响应拆包。
当前阶段只处理最小可用的 JSON 接口，不负责流式代理和长连接管理。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import ConfigurationException, UpstreamServiceException

DEFAULT_RAGFLOW_TIMEOUT_SECONDS = 15.0


class RagflowClient:
    """RAGFlow HTTP 基础客户端。"""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._settings = get_settings()
        self._http_client = http_client

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        expect_envelope: bool = True,
    ) -> Any:
        """发送 RAGFlow 请求并返回解析后的数据负载。"""

        api_key = self._settings.ragflow_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationException(
                "未配置 RAGFLOW_API_KEY，无法调用知识库。",
                details={"config_key": "RAGFLOW_API_KEY"},
            )

        request_headers = {
            "Authorization": f"Bearer {api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        normalized_params = self._drop_none_values(params)
        normalized_json_body = self._drop_none_values(json_body)

        try:
            if self._http_client is not None:
                response = await self._http_client.request(
                    method=method,
                    url=path,
                    params=normalized_params,
                    json=normalized_json_body,
                    headers=request_headers,
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._settings.ragflow_base_url.rstrip("/"),
                    timeout=DEFAULT_RAGFLOW_TIMEOUT_SECONDS,
                ) as http_client:
                    response = await http_client.request(
                        method=method,
                        url=path,
                        params=normalized_params,
                        json=normalized_json_body,
                        headers=request_headers,
                    )
            response.raise_for_status()
        except httpx.HTTPStatusError as exception:
            raise UpstreamServiceException(
                "RAGFlow 返回了非成功状态码。",
                error_code="ragflow_http_error",
                status_code=exception.response.status_code,
                details={"path": path, "response_text": exception.response.text},
            ) from exception
        except httpx.HTTPError as exception:
            raise UpstreamServiceException(
                "调用 RAGFlow 失败，请检查网络或服务地址。",
                error_code="ragflow_connection_error",
                details={"path": path},
            ) from exception

        try:
            response_payload = response.json()
        except ValueError as exception:
            raise UpstreamServiceException(
                "RAGFlow 返回了无法解析的 JSON。",
                error_code="ragflow_invalid_response",
                details={"path": path},
            ) from exception

        if not expect_envelope:
            return response_payload
        return self._extract_envelope_data(response_payload, path=path)

    @staticmethod
    def _drop_none_values(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """移除顶层为 None 的字段，避免把空参数错误传给 RAGFlow。"""

        if payload is None:
            return None
        normalized_payload = {
            str(field_name): field_value
            for field_name, field_value in payload.items()
            if field_value is not None
        }
        return normalized_payload or None

    @staticmethod
    def _extract_envelope_data(response_payload: Any, *, path: str) -> Any:
        """解析 RAGFlow 通用响应包，提取 data 字段。"""

        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "RAGFlow 返回了意外的响应结构。",
                error_code="ragflow_invalid_response",
                details={"path": path},
            )

        response_code = response_payload.get("code", 0)
        if response_code not in {0, 200}:
            raise UpstreamServiceException(
                str(response_payload.get("message") or "RAGFlow 返回了业务错误。"),
                error_code="ragflow_business_error",
                details={"path": path, "response": response_payload},
            )

        return response_payload.get("data", response_payload)
