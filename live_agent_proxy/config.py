"""Configuration constants for the monitor-network proxy routes."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import MONITOR_NETWORK_BRIDGE_BASE_URL, get_settings

MONITOR_NETWORK_PROXY_TIMEOUT_SECONDS = 30.0
MONITOR_NETWORK_LIVE_AGENT_BASE_URL = "http://33.69.9.33:8081"
MONITOR_NETWORK_LLM_BASE_URL = "http://12.1.90.211:32788"
MONITOR_NETWORK_RAGFLOW_BASE_URL = "http://33.69.3.30:8008"
MONITOR_NETWORK_RAGFLOW_API_KEY = (
    "ragflow-He4c0XmA3c52-O5DNg9Jup2XM0TrDO_vO_zKSfDAxzc"
)


@dataclass(frozen=True)
class ProxySettings:
    """Settings for the transparent proxy service."""

    proxy_app_name: str = "live-agent-proxy"
    proxy_timeout_seconds: float = MONITOR_NETWORK_PROXY_TIMEOUT_SECONDS
    upstream_base_url: str = MONITOR_NETWORK_LIVE_AGENT_BASE_URL
    llm_upstream_base_url: str = MONITOR_NETWORK_LLM_BASE_URL
    ragflow_upstream_base_url: str = MONITOR_NETWORK_RAGFLOW_BASE_URL
    ragflow_api_key: str = MONITOR_NETWORK_RAGFLOW_API_KEY


def get_proxy_settings() -> ProxySettings:
    """Return proxy settings backed by module constants."""

    app_settings = get_settings()
    if app_settings.enable_monitor_network_proxy and app_settings.is_debug:
        bridge_base_url = MONITOR_NETWORK_BRIDGE_BASE_URL.rstrip("/")
        return ProxySettings(
            upstream_base_url=bridge_base_url,
            llm_upstream_base_url=bridge_base_url,
            ragflow_upstream_base_url=f"{bridge_base_url}/ragflow",
        )
    return ProxySettings()
