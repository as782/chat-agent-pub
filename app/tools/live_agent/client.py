"""直播问答接口客户端模块。

负责封装直播问答智能体文档中的 HTTP 接口，并提供统一的请求发送与响应拆包能力。
当前阶段优先支持路线、路况、服务区和整体路网四类查询，不负责复杂鉴权和重试编排。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import UpstreamServiceException

DRIVING_PATH = "/agent/driving"
EVENT_PATH = "/agent/event"
SERVICE_PATH = "/agent/service"
NETWORK_OVERVIEW_PATH = "/agent/network-overview"


class LiveAgentClient:
    """直播问答接口 HTTP 客户端。"""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._settings = get_settings()
        self._http_client = http_client

    async def query_driving_plan(self, *, start: str, end: str) -> dict[str, Any]:
        """查询路线规划结果。"""

        response_payload = await self.request(
            "GET",
            DRIVING_PATH,
            params={"start": start, "end": end},
        )
        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "路线查询接口返回了意外的响应结构。",
                error_code="live_agent_invalid_response",
                details={"path": DRIVING_PATH},
            )
        return response_payload

    async def query_road_events(self, *, road: str) -> list[dict[str, Any]]:
        """查询指定道路的路况事件。"""

        response_payload = await self.request(
            "GET",
            EVENT_PATH,
            params={"road": road},
        )
        if not isinstance(response_payload, list):
            raise UpstreamServiceException(
                "路况查询接口返回了意外的响应结构。",
                error_code="live_agent_invalid_response",
                details={"path": EVENT_PATH},
            )
        return [item for item in response_payload if isinstance(item, dict)]

    async def query_services(self, *, keyword: str) -> list[dict[str, Any]]:
        """查询服务区相关信息。"""

        response_payload = await self.request(
            "GET",
            SERVICE_PATH,
            params={"keyword": keyword},
        )
        if not isinstance(response_payload, list):
            raise UpstreamServiceException(
                "服务区查询接口返回了意外的响应结构。",
                error_code="live_agent_invalid_response",
                details={"path": SERVICE_PATH},
            )
        return [item for item in response_payload if isinstance(item, dict)]

    async def query_network_overview(
        self,
        *,
        scope: str,
        query: str,
        report_type: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """查询整体路网概况。"""

        return await self.request(
            "GET",
            NETWORK_OVERVIEW_PATH,
            params={
                "scope": scope,
                "query": query,
                "report_type": report_type,
            },
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """发送直播问答接口请求并返回 data 字段。"""

        normalized_params = self._drop_none_values(params)
        try:
            if self._http_client is not None:
                response = await self._http_client.request(
                    method=method,
                    url=path,
                    params=normalized_params,
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._settings.live_agent_base_url.rstrip("/"),
                    timeout=self._settings.live_agent_timeout_seconds,
                ) as http_client:
                    response = await http_client.request(
                        method=method,
                        url=path,
                        params=normalized_params,
                    )
            response.raise_for_status()
        except httpx.HTTPStatusError as exception:
            raise UpstreamServiceException(
                "直播问答接口返回了非成功状态码。",
                error_code="live_agent_http_error",
                status_code=exception.response.status_code,
                details={"path": path, "response_text": exception.response.text},
            ) from exception
        except httpx.HTTPError as exception:
            raise UpstreamServiceException(
                "调用直播问答接口失败，请检查服务地址或网络。",
                error_code="live_agent_connection_error",
                details={"path": path},
            ) from exception

        try:
            response_payload = response.json()
        except ValueError as exception:
            raise UpstreamServiceException(
                "直播问答接口返回了无法解析的 JSON。",
                error_code="live_agent_invalid_response",
                details={"path": path},
            ) from exception

        return self._extract_envelope_data(response_payload, path=path)

    @staticmethod
    def _drop_none_values(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """移除顶层值为 None 的请求参数。"""

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
        """解析通用响应包并提取 data 字段。"""

        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "直播问答接口返回了意外的响应结构。",
                error_code="live_agent_invalid_response",
                details={"path": path},
            )

        response_code = response_payload.get("code", 0)
        if response_code not in {0, 200}:
            raise UpstreamServiceException(
                str(response_payload.get("message") or "直播问答接口返回了业务错误。"),
                error_code="live_agent_business_error",
                details={"path": path, "response": response_payload},
            )
        return response_payload.get("data", response_payload)
