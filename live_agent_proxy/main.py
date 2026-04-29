"""Transparent proxy for live-agent HTTP endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from live_agent_proxy.config import ProxySettings, get_proxy_settings

PROXIED_PATHS = (
    "/agent/driving",
    "/agent/event",
    "/agent/service",
    "/agent/topN",
)
RAGFLOW_PROXY_PREFIX = "/ragflow"
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
LLM_PROXY_PATH = "/v1/chat/monitor-completions"
LLM_UPSTREAM_PATH = "/v1/chat/completions"
PROXY_STATE_HTTP_CLIENT_KEY = "monitor_network_proxy_http_client"
proxy_router = APIRouter(tags=["monitor-network-proxy"])


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _build_upstream_url(base_url: str, path: str, raw_query: bytes) -> str:
    normalized_base_url = base_url.rstrip("/")
    if raw_query:
        return f"{normalized_base_url}{path}?{raw_query.decode('utf-8')}"
    return f"{normalized_base_url}{path}"


def build_proxy_http_client() -> httpx.AsyncClient:
    """Create the shared HTTP client used by proxy routes."""

    settings = get_proxy_settings()
    return httpx.AsyncClient(timeout=httpx.Timeout(settings.proxy_timeout_seconds))


def _resolve_http_client(request: Request) -> httpx.AsyncClient:
    http_client = getattr(request.app.state, PROXY_STATE_HTTP_CLIENT_KEY, None)
    if isinstance(http_client, httpx.AsyncClient):
        return http_client
    raise HTTPException(status_code=503, detail="Monitor-network proxy is not initialized.")


def _build_request_headers(
    request: Request,
    *,
    default_authorization: str | None = None,
) -> dict[str, str]:
    forwarded_headers: dict[str, str] = {}
    for header_name in ("accept", "authorization", "content-type"):
        header_value = request.headers.get(header_name)
        if header_value:
            forwarded_headers[header_name] = header_value
    if "authorization" not in forwarded_headers and default_authorization is not None:
        forwarded_headers["authorization"] = default_authorization
    return forwarded_headers


async def _proxy_request(
    request: Request,
    *,
    upstream_url: str,
    default_authorization: str | None = None,
) -> Response:
    http_client = _resolve_http_client(request)
    request_body = await request.body()

    try:
        upstream_request = http_client.build_request(
            method=request.method,
            url=upstream_url,
            headers=_build_request_headers(
                request,
                default_authorization=default_authorization,
            ),
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


async def _proxy_get(request: Request, path: str) -> Response:
    settings = get_proxy_settings()
    upstream_url = _build_upstream_url(
        settings.upstream_base_url,
        path,
        request.url.query.encode("utf-8"),
    )
    return await _proxy_request(request, upstream_url=upstream_url)


async def _proxy_chat_completions(request: Request) -> Response:
    settings = get_proxy_settings()
    upstream_url = f"{settings.llm_upstream_base_url.rstrip('/')}{LLM_UPSTREAM_PATH}"
    return await _proxy_request(request, upstream_url=upstream_url)


async def _proxy_ragflow_request(request: Request, path: str = "") -> Response:
    settings = get_proxy_settings()
    normalized_path = f"/{path.lstrip('/')}" if path else ""
    upstream_url = _build_upstream_url(
        settings.ragflow_upstream_base_url,
        normalized_path,
        request.url.query.encode("utf-8"),
    )
    default_authorization = (
        f"Bearer {settings.ragflow_api_key}" if settings.ragflow_api_key else None
    )
    return await _proxy_request(
        request,
        upstream_url=upstream_url,
        default_authorization=default_authorization,
    )


def _register_proxy_routes(router: APIRouter, paths: Iterable[str]) -> None:
    for path in paths:
        async def handler(request: Request, proxied_path: str = path) -> Response:
            return await _proxy_get(request, proxied_path)

        router.add_api_route(path, handler, methods=["GET"])


_register_proxy_routes(proxy_router, PROXIED_PATHS)
proxy_router.add_api_route(
    LLM_PROXY_PATH,
    _proxy_chat_completions,
    methods=["POST"],
)
proxy_router.add_api_route(
    RAGFLOW_PROXY_PREFIX,
    _proxy_ragflow_request,
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
proxy_router.add_api_route(
    f"{RAGFLOW_PROXY_PREFIX}/{{path:path}}",
    _proxy_ragflow_request,
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.monitor_network_proxy_http_client = build_proxy_http_client()
    yield
    await application.state.monitor_network_proxy_http_client.aclose()


def create_app() -> FastAPI:
    settings = get_proxy_settings()
    application = FastAPI(
        title=settings.proxy_app_name,
        version="0.1.0",
        description="Transparent proxy for monitor-network dependencies.",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    application.include_router(proxy_router)
    return application


app = create_app()
