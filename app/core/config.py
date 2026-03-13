"""配置模块。

负责从环境变量读取应用配置，并向全局提供统一的配置访问入口。
当前阶段不负责配置中心接入、动态刷新和密钥托管。
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置对象。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="chat-agent-backend", validation_alias="APP_NAME")
    app_env: str = Field(default="local", validation_alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=8000, validation_alias="APP_PORT")
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://postgres:postgres@postgres:5432/chat_agent",
        validation_alias="POSTGRES_DSN",
    )
    redis_url: str = Field(default="redis://redis:6379/0", validation_alias="REDIS_URL")
    ragflow_base_url: str = Field(
        default="http://ragflow:9380",
        validation_alias="RAGFLOW_BASE_URL",
    )
    ragflow_api_key: SecretStr | None = Field(default=None, validation_alias="RAGFLOW_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    @property
    def is_debug(self) -> bool:
        """根据运行环境判断是否启用调试级日志。"""

        return self.app_env.lower() in {"local", "dev", "test"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取缓存后的应用配置。

    使用缓存是为了避免在一次请求链路中重复读取环境变量，
    同时也方便测试时显式清空缓存后重新装载配置。
    """

    return Settings()
