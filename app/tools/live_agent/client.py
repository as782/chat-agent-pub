"""HTTP client for live-agent tools."""

from __future__ import annotations

from collections.abc import Mapping
from json import dumps
from time import perf_counter
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import UpstreamServiceException
from app.core.logger import get_logger

DRIVING_PATH = "/agent/driving"
EVENT_PATH = "/agent/event"
SERVICE_PATH = "/agent/service"
NETWORK_OVERVIEW_PATH = "/agent/topN"
MAX_LOGGED_RESPONSE_LENGTH = 4000

LOGGER = get_logger(__name__)


class LiveAgentClient:
    """HTTP client for live-agent endpoints."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._settings = get_settings()
        self._http_client = http_client

    async def query_driving_plan(self, *, start: str, end: str) -> dict[str, Any]:
        response_payload = await self.request(
            "GET",
            DRIVING_PATH,
            params={"start": start, "end": end},
        )
        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "Route planning endpoint returned an unexpected response structure.",
                error_code="live_agent_invalid_response",
                details={"path": DRIVING_PATH},
            )
        return response_payload

    async def query_road_events(self, *, road: str) -> list[dict[str, Any]]:
        response_payload = await self.request(
            "GET",
            EVENT_PATH,
            params={"road": road},
        )
        if not isinstance(response_payload, list):
            raise UpstreamServiceException(
                "Road event endpoint returned an unexpected response structure.",
                error_code="live_agent_invalid_response",
                details={"path": EVENT_PATH},
            )
        return [item for item in response_payload if isinstance(item, dict)]

    async def query_services(self, *, keyword: str) -> list[dict[str, Any]]:
        response_payload = await self.request(
            "GET",
            SERVICE_PATH,
            params={"keyword": keyword},
        )
        if not isinstance(response_payload, list):
            raise UpstreamServiceException(
                "Service endpoint returned an unexpected response structure.",
                error_code="live_agent_invalid_response",
                details={"path": SERVICE_PATH},
            )
        return [item for item in response_payload if isinstance(item, dict)]

    async def query_network_overview(self) -> dict[str, Any]:
        response_payload = await self.request("GET", NETWORK_OVERVIEW_PATH)
        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "Network overview endpoint returned an unexpected response structure.",
                error_code="live_agent_invalid_response",
                details={"path": NETWORK_OVERVIEW_PATH},
            )
        return response_payload

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        normalized_params = self._drop_none_values(params)
        connect_timeout_seconds = self._settings.live_agent_timeout_seconds
        request_start_time = perf_counter()
        base_url = self._resolve_base_url()

        LOGGER.info(
            (
                "Live Agent request started: method=%s path=%s base_url=%s "
                "connect_timeout_seconds=%.2f params=%s"
            ),
            method,
            path,
            base_url,
            connect_timeout_seconds,
            normalized_params,
        )

        try:
            if self._http_client is not None:
                response = await self._http_client.request(
                    method=method,
                    url=path,
                    params=normalized_params,
                    timeout=self._build_http_timeout(connect_timeout_seconds),
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._settings.live_agent_base_url.rstrip("/"),
                    timeout=self._build_http_timeout(connect_timeout_seconds),
                ) as http_client:
                    response = await http_client.request(
                        method=method,
                        url=path,
                        params=normalized_params,
                        timeout=self._build_http_timeout(connect_timeout_seconds),
                    )
            response.raise_for_status()
            try:
                response_payload = response.json()
            except ValueError as exception:
                LOGGER.warning(
                    (
                        "Live Agent response parsing failed: method=%s path=%s base_url=%s "
                        "duration_ms=%.2f connect_timeout_seconds=%.2f response_text=%s"
                    ),
                    method,
                    path,
                    base_url,
                    (perf_counter() - request_start_time) * 1000,
                    connect_timeout_seconds,
                    self._serialize_for_logging(response.text),
                )
                raise UpstreamServiceException(
                    "Live-agent endpoint returned invalid JSON.",
                    error_code="live_agent_invalid_response",
                    details={"path": path},
                ) from exception
        except httpx.HTTPStatusError as exception:
            LOGGER.warning(
                (
                    "Live Agent request failed: method=%s path=%s base_url=%s duration_ms=%.2f "
                    "connect_timeout_seconds=%.2f status_code=%s response_text=%s"
                ),
                method,
                path,
                base_url,
                (perf_counter() - request_start_time) * 1000,
                connect_timeout_seconds,
                exception.response.status_code,
                self._serialize_for_logging(exception.response.text),
            )
            raise UpstreamServiceException(
                "Live-agent endpoint returned a non-success status code.",
                error_code="live_agent_http_error",
                status_code=exception.response.status_code,
                details={"path": path, "response_text": exception.response.text},
            ) from exception
        except httpx.HTTPError as exception:
            LOGGER.warning(
                (
                    "Live Agent request failed: method=%s path=%s base_url=%s duration_ms=%.2f "
                    "connect_timeout_seconds=%.2f error_type=%s"
                ),
                method,
                path,
                base_url,
                (perf_counter() - request_start_time) * 1000,
                connect_timeout_seconds,
                type(exception).__name__,
            )
            raise UpstreamServiceException(
                "Failed to call live-agent endpoint. Please check the service address or network.",
                error_code="live_agent_connection_error",
                details={"path": path},
            ) from exception

        LOGGER.info(
            (
                "Live Agent response received: method=%s path=%s base_url=%s "
                "status_code=%s response_payload=%s"
            ),
            method,
            path,
            base_url,
            response.status_code,
            self._serialize_for_logging(response_payload),
        )
        LOGGER.info(
            (
                "Live Agent request completed: method=%s path=%s base_url=%s status_code=%s "
                "duration_ms=%.2f connect_timeout_seconds=%.2f"
            ),
            method,
            path,
            base_url,
            response.status_code,
            (perf_counter() - request_start_time) * 1000,
            connect_timeout_seconds,
        )
        return self._extract_envelope_data(response_payload, path=path)

    def _resolve_base_url(self) -> str:
        if self._http_client is not None:
            injected_base_url = str(getattr(self._http_client, "base_url", "")).rstrip("/")
            return injected_base_url or "injected-client"
        return self._settings.live_agent_base_url.rstrip("/")

    @staticmethod
    def _build_http_timeout(connect_timeout_seconds: float) -> httpx.Timeout:
        return httpx.Timeout(None, connect=connect_timeout_seconds)

    @staticmethod
    def _drop_none_values(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        normalized_payload = {
            str(field_name): field_value
            for field_name, field_value in payload.items()
            if field_value is not None
        }
        return normalized_payload or None

    @staticmethod
    def _serialize_for_logging(payload: Any) -> str:
        if isinstance(payload, str):
            normalized_payload = payload
        else:
            normalized_payload = dumps(payload, ensure_ascii=False, default=str)
        if len(normalized_payload) <= MAX_LOGGED_RESPONSE_LENGTH:
            return normalized_payload
        return normalized_payload[:MAX_LOGGED_RESPONSE_LENGTH] + " ...<truncated>"

    @staticmethod
    def _extract_envelope_data(response_payload: Any, *, path: str) -> Any:
        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "Live-agent endpoint returned an unexpected response structure.",
                error_code="live_agent_invalid_response",
                details={"path": path},
            )

        response_code = response_payload.get("code", 0)
        if response_code not in {0, 200}:
            raise UpstreamServiceException(
                str(
                    response_payload.get("message")
                    or "Live-agent endpoint returned a business error."
                ),
                error_code="live_agent_business_error",
                details={"path": path, "response": response_payload},
            )
        return response_payload.get("data", response_payload)
