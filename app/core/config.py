"""配置模块。

负责从环境变量读取应用配置，并向全局提供统一的配置访问入口。
当前阶段不负责配置中心接入、动态刷新和密钥托管。
"""

from functools import lru_cache
from urllib.parse import quote_plus

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

    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=55432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="chat_agent", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", validation_alias="POSTGRES_USER")
    postgres_password: SecretStr = Field(
        default=SecretStr("postgres"),
        validation_alias="POSTGRES_PASSWORD",
    )
    postgres_dsn: str | None = Field(default=None, validation_alias="POSTGRES_DSN")

    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    ragflow_base_url: str = Field(
        default="http://ragflow:9380",
        validation_alias="RAGFLOW_BASE_URL",
    )
    ragflow_api_key: SecretStr | None = Field(default=None, validation_alias="RAGFLOW_API_KEY")
    default_knowledge_dataset_id: str | None = Field(
        default=None,
        validation_alias="DEFAULT_KNOWLEDGE_DATASET_ID",
    )
    mcp_servers_json: str | None = Field(default=None, validation_alias="MCP_SERVERS_JSON")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    planner_use_llm: bool = Field(default=False, validation_alias="PLANNER_USE_LLM")
    planner_model: str | None = Field(default=None, validation_alias="PLANNER_MODEL")

    @property
    def database_url(self) -> str:
        """返回最终使用的数据库连接串。

        优先使用显式配置的 DSN。
        如果未配置 DSN，则根据主机、端口、账号和数据库名组装。
        这样可以同时兼容宿主机直连和 Docker 容器内服务发现。
        """

        if self.postgres_dsn:
            return self.postgres_dsn

        encoded_password = quote_plus(self.postgres_password.get_secret_value())
        return (
            "postgresql+asyncpg://"
            f"{self.postgres_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

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
