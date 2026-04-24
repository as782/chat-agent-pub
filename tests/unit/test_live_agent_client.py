"""Unit tests for the live-agent client."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from app.core.exceptions import UpstreamServiceException
from app.tools.live_agent.client import LiveAgentClient


@pytest.fixture(autouse=True)
def disable_terminal_exec_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing direct-call tests isolated from a local development .env."""

    monkeypatch.setenv("LIVE_AGENT_TERMINAL_EXEC_ENABLED", "false")


@pytest.mark.asyncio
async def test_live_agent_client_logs_success_response_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Successful responses should log the returned payload."""

    async def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"code": 0, "data": {"queryTime": "2026-04-18 10:00:00", "routesCount": 2}},
        )

    caplog.set_level(logging.INFO, logger="app.tools.live_agent.client")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
        base_url="https://example.com",
    ) as http_client:
        client = LiveAgentClient(http_client=http_client)
        result = await client.request("GET", "/agent/topN")

    assert result == {"queryTime": "2026-04-18 10:00:00", "routesCount": 2}
    assert "Live Agent response received:" in caplog.text
    assert '"code": 0' in caplog.text
    assert '"routesCount": 2' in caplog.text


@pytest.mark.asyncio
async def test_live_agent_client_logs_business_error_response_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Business-error payloads should still be logged before raising."""

    async def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"code": 5001, "message": "upstream busy", "data": {"road": "G60"}},
        )

    caplog.set_level(logging.INFO, logger="app.tools.live_agent.client")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
        base_url="https://example.com",
    ) as http_client:
        client = LiveAgentClient(http_client=http_client)
        with pytest.raises(UpstreamServiceException, match="upstream busy"):
            await client.request("GET", "/agent/event")

    assert "Live Agent response received:" in caplog.text
    assert '"message": "upstream busy"' in caplog.text
    assert '"road": "G60"' in caplog.text


@pytest.mark.asyncio
async def test_live_agent_client_logs_http_error_response_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-2xx responses should log the raw response text."""

    async def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=502,
            text='{"code":502,"message":"bad gateway"}',
            headers={"Content-Type": "application/json"},
        )

    caplog.set_level(logging.WARNING, logger="app.tools.live_agent.client")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
        base_url="https://example.com",
    ) as http_client:
        client = LiveAgentClient(http_client=http_client)
        with pytest.raises(UpstreamServiceException, match="non-success status code"):
            await client.request("GET", "/agent/service")

    assert "Live Agent request failed:" in caplog.text
    assert 'response_text={"code":502,"message":"bad gateway"}' in caplog.text


@pytest.mark.asyncio
async def test_live_agent_client_logs_invalid_json_response_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid JSON responses should log the raw response text."""

    async def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, text="not-json-response")

    caplog.set_level(logging.WARNING, logger="app.tools.live_agent.client")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
        base_url="https://example.com",
    ) as http_client:
        client = LiveAgentClient(http_client=http_client)
        with pytest.raises(UpstreamServiceException, match="invalid JSON"):
            await client.request("GET", "/agent/driving")

    assert "Live Agent response parsing failed:" in caplog.text
    assert "response_text=not-json-response" in caplog.text


@pytest.mark.asyncio
async def test_live_agent_client_uses_terminal_exec_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Development proxy mode should call terminal_exec and parse the inner live-agent envelope."""

    monkeypatch.setenv("LIVE_AGENT_TERMINAL_EXEC_ENABLED", "true")
    monkeypatch.setenv("LIVE_AGENT_TERMINAL_EXEC_URL", "https://proxy.example/terminal_exec")
    monkeypatch.setenv("LIVE_AGENT_TERMINAL_TARGET_BASE_URL", "http://33.69.9.33:8081")

    captured: dict[str, str] = {}

    async def fake_call_terminal_exec(
        self: LiveAgentClient,
        *,
        command: str,
    ) -> dict[str, Any]:
        del self
        captured["command"] = command
        return {
            "data": {
                "success": True,
                "output": (
                    '{"code":200,"message":"操作成功",'
                    '"data":{"routesCount":1,"routes":[]}}'
                ),
            }
        }

    monkeypatch.setattr(LiveAgentClient, "_call_terminal_exec", fake_call_terminal_exec)

    client = LiveAgentClient()
    result = await client.request(
        "GET",
        "/agent/driving",
        params={"start": "温州", "end": "金华"},
    )

    assert result == {"routesCount": 1, "routes": []}
    assert captured["command"] == (
        "curl -sS "
        "'http://33.69.9.33:8081/agent/driving?"
        "start=%E6%B8%A9%E5%B7%9E&end=%E9%87%91%E5%8D%8E'"
    )


@pytest.mark.asyncio
async def test_live_agent_client_terminal_exec_normalizes_null_event_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The monitor-network event endpoint can return data=null for no events."""

    monkeypatch.setenv("LIVE_AGENT_TERMINAL_EXEC_ENABLED", "true")
    monkeypatch.setenv("LIVE_AGENT_TERMINAL_TARGET_BASE_URL", "http://33.69.9.33:8081")

    async def fake_call_terminal_exec(
        self: LiveAgentClient,
        *,
        command: str,
    ) -> dict[str, Any]:
        del self, command
        return {
            "data": {
                "success": True,
                "output": '{"code":200,"message":"操作成功","data":null}',
            }
        }

    monkeypatch.setattr(LiveAgentClient, "_call_terminal_exec", fake_call_terminal_exec)

    client = LiveAgentClient()
    result = await client.query_road_events(road="沪杭高速")

    assert result == []
