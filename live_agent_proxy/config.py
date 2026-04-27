"""Configuration constants for the monitor-network proxy routes."""

from __future__ import annotations

from dataclasses import dataclass

MONITOR_NETWORK_PROXY_TIMEOUT_SECONDS = 30.0
MONITOR_NETWORK_LIVE_AGENT_BASE_URL = "http://33.69.9.33:8081"
MONITOR_NETWORK_LLM_BASE_URL = "http://12.1.90.211:32788"


@dataclass(frozen=True)
class ProxySettings:
    """Settings for the transparent proxy service."""

    proxy_app_name: str = "live-agent-proxy"
    proxy_timeout_seconds: float = MONITOR_NETWORK_PROXY_TIMEOUT_SECONDS
    upstream_base_url: str = MONITOR_NETWORK_LIVE_AGENT_BASE_URL
    llm_upstream_base_url: str = MONITOR_NETWORK_LLM_BASE_URL


def get_proxy_settings() -> ProxySettings:
    """Return proxy settings backed by module constants."""

    return ProxySettings()
