"""HTTP client for RAGFlow APIs."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import ConfigurationException, UpstreamServiceException
from app.core.logger import get_logger

LOGGER = get_logger(__name__)


class RagflowClient:
    """HTTP client for RAGFlow."""

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
        api_key = self._settings.ragflow_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationException(
                "RAGFLOW_API_KEY is not configured, so RAGFlow cannot be called.",
                details={"config_key": "RAGFLOW_API_KEY"},
            )

        connect_timeout_seconds = self._settings.ragflow_timeout_seconds
        request_headers = {
            "Authorization": f"Bearer {api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        normalized_params = self._drop_none_values(params)
        normalized_json_body = self._drop_none_values(json_body)
        request_start_time = perf_counter()
        base_url = self._resolve_base_url()

        LOGGER.info(
            (
                "RAGFlow request started: method=%s path=%s base_url=%s connect_timeout_seconds=%.2f "
                "params=%s json_body_keys=%s expect_envelope=%s"
            ),
            method,
            path,
            base_url,
            connect_timeout_seconds,
            normalized_params,
            sorted(normalized_json_body.keys()) if normalized_json_body else [],
            expect_envelope,
        )

        try:
            if self._http_client is not None:
                response = await self._http_client.request(
                    method=method,
                    url=path,
                    params=normalized_params,
                    json=normalized_json_body,
                    headers=request_headers,
                    timeout=self._build_http_timeout(connect_timeout_seconds),
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._settings.ragflow_base_url.rstrip("/"),
                    timeout=self._build_http_timeout(connect_timeout_seconds),
                ) as http_client:
                    response = await http_client.request(
                        method=method,
                        url=path,
                        params=normalized_params,
                        json=normalized_json_body,
                        headers=request_headers,
                        timeout=self._build_http_timeout(connect_timeout_seconds),
                    )
            response.raise_for_status()
        except httpx.HTTPStatusError as exception:
            LOGGER.warning(
                (
                    "RAGFlow request failed: method=%s path=%s base_url=%s duration_ms=%.2f "
                    "connect_timeout_seconds=%.2f status_code=%s"
                ),
                method,
                path,
                base_url,
                (perf_counter() - request_start_time) * 1000,
                connect_timeout_seconds,
                exception.response.status_code,
            )
            raise UpstreamServiceException(
                "RAGFlow returned a non-success status code.",
                error_code="ragflow_http_error",
                status_code=exception.response.status_code,
                details={"path": path, "response_text": exception.response.text},
            ) from exception
        except httpx.HTTPError as exception:
            LOGGER.warning(
                (
                    "RAGFlow request failed: method=%s path=%s base_url=%s duration_ms=%.2f "
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
                "Failed to call RAGFlow. Please check the network or service address.",
                error_code="ragflow_connection_error",
                details={"path": path},
            ) from exception

        try:
            response_payload = response.json()
        except ValueError as exception:
            LOGGER.warning(
                (
                    "RAGFlow response parsing failed: method=%s path=%s base_url=%s "
                    "duration_ms=%.2f connect_timeout_seconds=%.2f"
                ),
                method,
                path,
                base_url,
                (perf_counter() - request_start_time) * 1000,
                connect_timeout_seconds,
            )
            raise UpstreamServiceException(
                "RAGFlow returned invalid JSON.",
                error_code="ragflow_invalid_response",
                details={"path": path},
            ) from exception

        LOGGER.info(
            (
                "RAGFlow request completed: method=%s path=%s base_url=%s status_code=%s "
                "duration_ms=%.2f connect_timeout_seconds=%.2f"
            ),
            method,
            path,
            base_url,
            response.status_code,
            (perf_counter() - request_start_time) * 1000,
            connect_timeout_seconds,
        )

        if not expect_envelope:
            return response_payload
        return self._extract_envelope_data(response_payload, path=path)

    def _resolve_base_url(self) -> str:
        if self._http_client is not None:
            injected_base_url = str(getattr(self._http_client, "base_url", "")).rstrip("/")
            return injected_base_url or "injected-client"
        return self._settings.ragflow_base_url.rstrip("/")

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
    def _extract_envelope_data(response_payload: Any, *, path: str) -> Any:
        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "RAGFlow returned an unexpected response structure.",
                error_code="ragflow_invalid_response",
                details={"path": path},
            )

        response_code = response_payload.get("code", 0)
        if response_code not in {0, 200}:
            raise UpstreamServiceException(
                str(response_payload.get("message") or "RAGFlow returned a business error."),
                error_code="ragflow_business_error",
                details={"path": path, "response": response_payload},
            )

        return response_payload.get("data", response_payload)
