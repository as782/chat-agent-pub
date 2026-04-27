"""Transparent proxy for live-agent HTTP endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from live_agent_proxy.config import ProxySettings, get_proxy_settings

PROXIED_PATHS = (
    "/agent/driving",
    "/agent/event",
    "/agent/service",
    "/agent/topN",
)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _build_upstream_url(settings: ProxySettings, path: str, raw_query: bytes) -> str:
    base_url = settings.upstream_base_url.rstrip("/")
    if raw_query:
        return f"{base_url}{path}?{raw_query.decode('utf-8')}"
    return f"{base_url}{path}"


async def _proxy_get(request: Request, path: str) -> Response:
    settings = get_proxy_settings()
    http_client: httpx.AsyncClient = request.app.state.http_client
    upstream_url = _build_upstream_url(settings, path, request.url.query.encode("utf-8"))

    try:
        upstream_response = await http_client.get(
            upstream_url,
            headers={"Accept": request.headers.get("accept", "*/*")},
        )
    except httpx.TimeoutException as exception:
        raise HTTPException(status_code=504, detail="Upstream request timed out.") from exception
    except httpx.HTTPError as exception:
        raise HTTPException(
            status_code=502,
            detail="Failed to reach upstream service.",
        ) from exception

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_filter_response_headers(upstream_response.headers),
    )


def _build_forwarded_headers(request: Request) -> dict[str, str]:
    forwarded_headers: dict[str, str] = {}
    for header_name in ("accept", "authorization", "content-type"):
        header_value = request.headers.get(header_name)
        if header_value:
            forwarded_headers[header_name] = header_value
    return forwarded_headers


async def _proxy_chat_completions(request: Request) -> Response:
    settings = get_proxy_settings()
    http_client: httpx.AsyncClient = request.app.state.http_client
    request_body = await request.body()
    upstream_url = f"{settings.llm_upstream_base_url.rstrip('/')}{CHAT_COMPLETIONS_PATH}"

    try:
        upstream_request = http_client.build_request(
            method="POST",
            url=upstream_url,
            headers=_build_forwarded_headers(request),
            content=request_body,
        )
        upstream_response = await http_client.send(upstream_request, stream=True)
    except httpx.TimeoutException as exception:
        raise HTTPException(status_code=504, detail="Upstream request timed out.") from exception
    except httpx.HTTPError as exception:
        raise HTTPException(
            status_code=502,
            detail="Failed to reach upstream service.",
        ) from exception

    response_headers = _filter_response_headers(upstream_response.headers)
    content_type = response_headers.get("content-type", "application/json")

    if "text/event-stream" in content_type.lower():
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=content_type,
            background=BackgroundTask(upstream_response.aclose),
        )

    try:
        response_content = await upstream_response.aread()
    finally:
        await upstream_response.aclose()

    return Response(
        content=response_content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_proxy_settings()
    application.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.proxy_timeout_seconds),
    )
    yield
    await application.state.http_client.aclose()


def create_app() -> FastAPI:
    settings = get_proxy_settings()
    application = FastAPI(
        title=settings.proxy_app_name,
        version="0.1.0",
        description="Transparent proxy for live-agent endpoints.",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    def _register_proxy_routes(paths: Iterable[str]) -> None:
        for path in paths:
            async def handler(request: Request, proxied_path: str = path) -> Response:
                return await _proxy_get(request, proxied_path)

            application.add_api_route(path, handler, methods=["GET"], tags=["proxy"])

    _register_proxy_routes(PROXIED_PATHS)
    application.add_api_route(
        CHAT_COMPLETIONS_PATH,
        _proxy_chat_completions,
        methods=["POST"],
        tags=["proxy"],
    )
    return application


app = create_app()
