"""HTTP client for live-agent tools."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from json import JSONDecodeError, dumps, loads
from time import perf_counter
from typing import Any
from urllib.parse import urlencode, urljoin

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
        use_terminal_exec = self._settings.live_agent_terminal_exec_enabled

        LOGGER.info(
            (
                "Live Agent request started: method=%s path=%s base_url=%s "
                "connect_timeout_seconds=%.2f via_terminal_exec=%s params=%s"
            ),
            method,
            path,
            base_url,
            connect_timeout_seconds,
            use_terminal_exec,
            normalized_params,
        )

        try:
            if use_terminal_exec:
                response_payload = await self._request_via_terminal_exec(
                    method=method,
                    path=path,
                    params=normalized_params,
                )
            else:
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
                "status_code=%s via_terminal_exec=%s response_payload=%s"
            ),
            method,
            path,
            base_url,
            200 if use_terminal_exec else response.status_code,
            use_terminal_exec,
            self._serialize_for_logging(response_payload),
        )
        LOGGER.info(
            (
                "Live Agent request completed: method=%s path=%s base_url=%s status_code=%s "
                "duration_ms=%.2f connect_timeout_seconds=%.2f via_terminal_exec=%s"
            ),
            method,
            path,
            base_url,
            200 if use_terminal_exec else response.status_code,
            (perf_counter() - request_start_time) * 1000,
            connect_timeout_seconds,
            use_terminal_exec,
        )
        return self._extract_envelope_data(response_payload, path=path)

    def _resolve_base_url(self) -> str:
        if self._settings.live_agent_terminal_exec_enabled:
            return self._resolve_terminal_target_base_url()
        if self._http_client is not None:
            injected_base_url = str(getattr(self._http_client, "base_url", "")).rstrip("/")
            return injected_base_url or "injected-client"
        return self._settings.live_agent_base_url.rstrip("/")

    def _resolve_terminal_target_base_url(self) -> str:
        return (
            self._settings.live_agent_terminal_target_base_url
            or self._settings.live_agent_base_url
        ).rstrip("/")

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

    async def _request_via_terminal_exec(
        self,
        *,
        method: str,
        path: str,
        params: Mapping[str, Any] | None,
    ) -> Any:
        command = self._build_terminal_curl_command(method=method, path=path, params=params)
        terminal_payload = await self._call_terminal_exec(command=command)
        output = self._extract_terminal_exec_output(terminal_payload)
        try:
            response_payload = loads(output)
        except JSONDecodeError as exception:
            LOGGER.warning(
                "Live Agent terminal_exec response parsing failed: path=%s output=%s",
                path,
                self._serialize_for_logging(output),
            )
            raise UpstreamServiceException(
                "terminal_exec returned invalid live-agent JSON.",
                error_code="live_agent_terminal_exec_invalid_response",
                details={"path": path},
            ) from exception

        if not isinstance(response_payload, dict):
            raise UpstreamServiceException(
                "terminal_exec returned an unexpected live-agent response structure.",
                error_code="live_agent_terminal_exec_invalid_response",
                details={"path": path},
            )

        return self._normalize_terminal_exec_payload(response_payload, path=path)

    def _build_terminal_curl_command(
        self,
        *,
        method: str,
        path: str,
        params: Mapping[str, Any] | None,
    ) -> str:
        base_url = self._resolve_terminal_target_base_url()
        target_url = urljoin(f"{base_url}/", path.lstrip("/"))
        if params:
            target_url = f"{target_url}?{urlencode(params)}"
        normalized_method = method.upper().strip()
        if normalized_method == "GET":
            return f"curl -sS {shlex.quote(target_url)}"
        return f"curl -sS -X {shlex.quote(normalized_method)} {shlex.quote(target_url)}"

    async def _call_terminal_exec(self, *, command: str) -> dict[str, Any]:
        payload = {
            "command": command,
            "timeout_seconds": self._settings.live_agent_terminal_exec_timeout_seconds,
        }
        last_output = ""
        for attempt in range(1, self._settings.live_agent_terminal_exec_retries + 1):
            try:
                raw_output = await self._run_terminal_exec_curl(payload)
                last_output = raw_output.strip()
                if not last_output:
                    raise UpstreamServiceException(
                        "terminal_exec returned an empty response.",
                        error_code="live_agent_terminal_exec_empty_response",
                    )
                terminal_payload = loads(last_output)
                if not isinstance(terminal_payload, dict):
                    raise UpstreamServiceException(
                        "terminal_exec returned an unexpected response structure.",
                        error_code="live_agent_terminal_exec_invalid_response",
                    )
                return terminal_payload
            except (JSONDecodeError, UpstreamServiceException) as exception:
                if attempt >= self._settings.live_agent_terminal_exec_retries:
                    if isinstance(exception, UpstreamServiceException):
                        raise
                    raise UpstreamServiceException(
                        "terminal_exec returned invalid JSON.",
                        error_code="live_agent_terminal_exec_invalid_response",
                        details={"response_text": last_output},
                    ) from exception
                await asyncio.sleep(min(attempt, 3))

        raise UpstreamServiceException(
            "terminal_exec request failed.",
            error_code="live_agent_terminal_exec_error",
        )

    async def _run_terminal_exec_curl(self, payload: dict[str, Any]) -> str:
        tmp_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        tmp_file_path = tmp_file.name
        try:
            tmp_file.write(dumps(payload, ensure_ascii=False, separators=(",", ":")))
            tmp_file.close()
            outer_timeout_seconds = (
                self._settings.live_agent_terminal_exec_timeout_seconds
                + self._settings.live_agent_timeout_seconds
            )
            try:
                completed_process = await asyncio.to_thread(
                    subprocess.run,
                    [
                        self._settings.live_agent_terminal_exec_curl_binary,
                        "-k",
                        "-sS",
                        "-X",
                        "POST",
                        self._settings.live_agent_terminal_exec_url,
                        "-H",
                        "Content-Type: application/json; charset=utf-8",
                        "--data-binary",
                        f"@{tmp_file_path}",
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=outer_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exception:
                raise UpstreamServiceException(
                    "terminal_exec curl request timed out.",
                    error_code="live_agent_terminal_exec_timeout",
                    details={"timeout_seconds": outer_timeout_seconds},
                ) from exception

            output = (completed_process.stdout or b"").decode("utf-8", errors="replace")
            if completed_process.returncode != 0:
                raise UpstreamServiceException(
                    "terminal_exec curl request failed.",
                    error_code="live_agent_terminal_exec_curl_error",
                    details={
                        "exit_code": completed_process.returncode,
                        "output": output,
                    },
                )
            return output
        except FileNotFoundError as exception:
            raise UpstreamServiceException(
                "terminal_exec curl binary was not found.",
                error_code="live_agent_terminal_exec_curl_not_found",
                details={"curl_binary": self._settings.live_agent_terminal_exec_curl_binary},
            ) from exception
        finally:
            if not tmp_file.closed:
                tmp_file.close()
            with suppress(OSError):
                os.remove(tmp_file_path)

    @staticmethod
    def _extract_terminal_exec_output(terminal_payload: dict[str, Any]) -> str:
        output: object | None = None
        success: object | None = terminal_payload.get("success")
        data = terminal_payload.get("data")
        if isinstance(data, dict):
            output = data.get("output")
            if "success" in data:
                success = data.get("success")
        elif "output" in terminal_payload:
            output = terminal_payload.get("output")

        if success is False:
            raise UpstreamServiceException(
                "terminal_exec command failed.",
                error_code="live_agent_terminal_exec_command_failed",
                details={"terminal_response": terminal_payload},
            )
        if not isinstance(output, str):
            raise UpstreamServiceException(
                "terminal_exec response did not include output.",
                error_code="live_agent_terminal_exec_invalid_response",
                details={"terminal_response": terminal_payload},
            )
        return output.strip()

    @staticmethod
    def _normalize_terminal_exec_payload(
        response_payload: dict[str, Any],
        *,
        path: str,
    ) -> dict[str, Any]:
        normalized_payload = dict(response_payload)
        if path in {EVENT_PATH, SERVICE_PATH} and normalized_payload.get("data") is None:
            normalized_payload["data"] = []
        return normalized_payload

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
