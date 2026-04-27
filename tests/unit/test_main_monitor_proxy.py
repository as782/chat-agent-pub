"""Tests for conditional monitor-network proxy route mounting."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pytest import MonkeyPatch

from app.core.config import get_settings


@pytest.fixture
def clear_app_settings_cache() -> Iterator[None]:
    """Ensure app settings are reloaded for each assertion."""

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_create_app_excludes_monitor_proxy_routes_by_default(
    monkeypatch: MonkeyPatch,
    clear_app_settings_cache: None,
) -> None:
    """Monitor proxy routes should not exist unless explicitly enabled."""

    monkeypatch.delenv("ENABLE_MONITOR_NETWORK_PROXY", raising=False)

    from app.main import create_app

    application = create_app()
    route_paths = {route.path for route in application.routes}

    assert "/agent/driving" not in route_paths
    assert "/v1/chat/monitor-completions" not in route_paths


def test_create_app_includes_monitor_proxy_routes_when_enabled(
    monkeypatch: MonkeyPatch,
    clear_app_settings_cache: None,
) -> None:
    """Monitor proxy routes should be mounted on the main FastAPI app when enabled."""

    monkeypatch.setenv("ENABLE_MONITOR_NETWORK_PROXY", "true")

    from app.main import create_app

    application = create_app()
    route_paths = {route.path for route in application.routes}

    assert "/agent/driving" in route_paths
    assert "/agent/event" in route_paths
    assert "/agent/service" in route_paths
    assert "/agent/topN" in route_paths
    assert "/v1/chat/monitor-completions" in route_paths
