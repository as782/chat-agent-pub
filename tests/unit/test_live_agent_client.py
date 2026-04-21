"""Unit tests for the live-agent client."""

from __future__ import annotations

import logging

import httpx
import pytest

from app.core.exceptions import UpstreamServiceException
from app.tools.live_agent.client import LiveAgentClient


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
