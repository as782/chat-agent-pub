"""Unit tests for the live-agent transparent proxy service."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from live_agent_proxy.config import get_proxy_settings
from live_agent_proxy.main import create_app

SUCCESS_PAYLOAD = (
    b'{"code":200,"message":"\xe6\x93\x8d\xe4\xbd\x9c\xe6\x88\x90\xe5\x8a\x9f",'
    b'"data":{"routesCount":1}}'
)


@pytest.fixture
def proxy_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client for the proxy app with clean settings cache."""

    get_proxy_settings.cache_clear()
    monkeypatch.setenv("LIVE_AGENT_UPSTREAM_BASE_URL", "http://upstream.example.com")
    monkeypatch.setenv("LLM_UPSTREAM_BASE_URL", "http://llm-upstream.example.com")
    app = create_app()
    with TestClient(app) as client:
        yield client
    get_proxy_settings.cache_clear()


def test_proxy_forwards_query_and_returns_upstream_json_unchanged(
    proxy_client: TestClient,
) -> None:
    """The proxy should forward the query string and return the exact upstream payload."""

    async def handle_request(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://upstream.example.com/agent/driving?start=%E6%9D%AD%E5%B7%9E&end=%E6%B8%A9%E5%B7%9E"
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8", "X-Upstream": "ok"},
            content=SUCCESS_PAYLOAD,
        )

    proxy_client.app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    )

    response = proxy_client.get("/agent/driving", params={"start": "杭州", "end": "温州"})

    assert response.status_code == 200
    assert response.content == SUCCESS_PAYLOAD
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.headers["x-upstream"] == "ok"


def test_proxy_returns_upstream_non_success_status_unchanged(proxy_client: TestClient) -> None:
    """The proxy should not rewrite upstream non-2xx responses."""

    async def handle_request(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://upstream.example.com/agent/event?road=G60"
        return httpx.Response(
            status_code=503,
            headers={"Content-Type": "application/json"},
            content=b'{"code":503,"message":"upstream busy","data":null}',
        )

    proxy_client.app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    )

    response = proxy_client.get("/agent/event", params={"road": "G60"})

    assert response.status_code == 503
    assert response.content == b'{"code":503,"message":"upstream busy","data":null}'


def test_proxy_returns_gateway_error_when_upstream_unreachable(proxy_client: TestClient) -> None:
    """Transport failures should surface as a gateway-style error."""

    async def handle_request(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    proxy_client.app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    )

    response = proxy_client.get("/agent/topN")

    assert response.status_code == 502
    assert response.json() == {"detail": "Failed to reach upstream service."}


def test_chat_completions_proxy_returns_non_stream_response_unchanged(
    proxy_client: TestClient,
) -> None:
    """The chat-completions proxy should pass through headers, body and response payload."""

    expected_request_body = {
        "model": "qwen3535ba3b",
        "messages": [{"role": "user", "content": "写一首关于春天的五言绝句"}],
        "stream": False,
    }

    async def handle_request(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://llm-upstream.example.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content.decode("utf-8")) == expected_request_body
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=(
                b'{"id":"chatcmpl-1","object":"chat.completion","choices":'
                b'[{"index":0,"message":{"role":"assistant","content":"\xe6\x98\xa5\xe9\xa3\x8e'
                b'\xe8\xbd\xbb\xe6\x8b\x82\xe6\x9f\xb3"},"finish_reason":"stop"}]}'
            ),
        )

    proxy_client.app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    )

    response = proxy_client.post(
        "/v1/chat/monitor-completions",
        headers={
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
        },
        json=expected_request_body,
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "春风轻拂柳"


def test_chat_completions_proxy_returns_stream_response_unchanged(
    proxy_client: TestClient,
) -> None:
    """The chat-completions proxy should preserve upstream SSE streaming output."""

    async def handle_request(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://llm-upstream.example.com/v1/chat/completions"
        assert json.loads(request.content.decode("utf-8"))["stream"] is True
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            stream=httpx.ByteStream(
                b'data: {"choices":[{"delta":{"content":"<think>"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"\xe6\x98\xa5\xe5\xb1\xb1\xe6\xb8\x90'
                b'\xe6\x9a\x96"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    proxy_client.app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    )

    with proxy_client.stream(
        "POST",
        "/v1/chat/monitor-completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3535ba3b",
            "messages": [{"role": "user", "content": "写一首关于春天的五言绝句"}],
            "stream": True,
        },
    ) as response:
        chunks = list(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "".join(chunks).endswith("data: [DONE]\n\n")
