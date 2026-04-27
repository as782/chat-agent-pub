"""Configuration for the live-agent transparent proxy service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProxySettings(BaseSettings):
    """Settings for the transparent proxy service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    proxy_app_name: str = Field(default="live-agent-proxy", validation_alias="PROXY_APP_NAME")
    proxy_app_host: str = Field(default="0.0.0.0", validation_alias="PROXY_APP_HOST")
    proxy_app_port: int = Field(default=8081, validation_alias="PROXY_APP_PORT")
    proxy_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="PROXY_TIMEOUT_SECONDS",
        gt=0,
    )
    upstream_base_url: str = Field(
        default="http://33.69.9.33:8081",
        validation_alias="LIVE_AGENT_UPSTREAM_BASE_URL",
    )
    llm_upstream_base_url: str = Field(
        default="http://12.1.90.211:32788",
        validation_alias="LLM_UPSTREAM_BASE_URL",
    )


@lru_cache(maxsize=1)
def get_proxy_settings() -> ProxySettings:
    """Return cached proxy settings."""

    return ProxySettings()
