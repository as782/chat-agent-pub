"""配置模块。

负责从环境变量读取应用配置，并向全局提供统一的配置访问入口。
当前阶段不负责配置中心接入、动态刷新和密钥托管。
"""

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MONITOR_NETWORK_BRIDGE_BASE_URL = "http://192.168.26.38:22372"
MONITOR_NETWORK_LIVE_AGENT_BASE_URL = MONITOR_NETWORK_BRIDGE_BASE_URL
MONITOR_NETWORK_OPENAI_BASE_URL = f"{MONITOR_NETWORK_BRIDGE_BASE_URL}/v1"
MONITOR_NETWORK_OPENAI_PROXY_PATH = "/v1/chat/monitor-completions"
MONITOR_NETWORK_OPENAI_API_KEY = "dKyDeqHfGLEAjSUQA46bFb6d1cBe4aA4822209012a8eF925"
MONITOR_NETWORK_OPENAI_MODEL = "qwen3535ba3b"
MONITOR_NETWORK_RAGFLOW_BASE_URL = f"{MONITOR_NETWORK_BRIDGE_BASE_URL}/ragflow"
MONITOR_NETWORK_RAGFLOW_API_KEY = (
    "ragflow-He4c0XmA3c52-O5DNg9Jup2XM0TrDO_vO_zKSfDAxzc"
)
MONITOR_NETWORK_DEFAULT_KNOWLEDGE_DATASET_ID = "a2e9f8ff324011f198edce511781c013"


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
    log_to_file: bool | None = Field(default=None, validation_alias="LOG_TO_FILE")
    log_dir: str = Field(default="logs", validation_alias="LOG_DIR")
    log_file_name: str = Field(default="chat-agent.log", validation_alias="LOG_FILE_NAME")
    log_rotate_when: str = Field(default="midnight", validation_alias="LOG_ROTATE_WHEN")
    log_rotate_interval: int = Field(default=1, validation_alias="LOG_ROTATE_INTERVAL", ge=1)
    log_backup_count: int = Field(default=14, validation_alias="LOG_BACKUP_COUNT", ge=0)

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
    live_agent_base_url: str = Field(
        default="http://localhost:8081",
        validation_alias="LIVE_AGENT_BASE_URL",
    )
    live_agent_timeout_seconds: float = Field(
        default=15.0,
        validation_alias="LIVE_AGENT_TIMEOUT_SECONDS",
    )
    enable_monitor_network_proxy: bool = Field(
        default=False,
        validation_alias="ENABLE_MONITOR_NETWORK_PROXY",
    )
    mcp_servers_json: str | None = Field(default=None, validation_alias="MCP_SERVERS_JSON")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_timeout_seconds: float = Field(
        default=60.0,
        validation_alias="OPENAI_TIMEOUT_SECONDS",
    )
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    openai_enable_thinking: bool | None = Field(
        default=None,
        validation_alias="OPENAI_ENABLE_THINKING",
    )
    planner_model: str | None = Field(default=None, validation_alias="PLANNER_MODEL")
    planner_base_url: str | None = Field(default=None, validation_alias="PLANNER_BASE_URL")
    planner_api_key: SecretStr | None = Field(default=None, validation_alias="PLANNER_API_KEY")
    planner_enable_thinking: bool | None = Field(
        default=None,
        validation_alias="PLANNER_ENABLE_THINKING",
    )
    planner_timeout_seconds: float | None = Field(
        default=None,
        validation_alias="PLANNER_TIMEOUT_SECONDS",
    )
    ragflow_timeout_seconds: float = Field(
        default=15.0,
        validation_alias="RAGFLOW_TIMEOUT_SECONDS",
    )
    mcp_http_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="MCP_HTTP_TIMEOUT_SECONDS",
    )
    mcp_sse_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="MCP_SSE_TIMEOUT_SECONDS",
    )
    mcp_sse_read_timeout_seconds: float = Field(
        default=60.0,
        validation_alias="MCP_SSE_READ_TIMEOUT_SECONDS",
    )

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

    @property
    def enable_file_logging(self) -> bool:
        """返回是否启用日志文件持久化。"""

        if self.log_to_file is not None:
            return self.log_to_file
        return self.app_env.lower() != "test"

    @property
    def use_monitor_network_development_upstreams(self) -> bool:
        """Whether local/dev runs should use the bridge-machine proxy endpoints."""

        return self.enable_monitor_network_proxy and self.is_debug

    @property
    def resolved_live_agent_base_url(self) -> str:
        """Return the effective live-agent base URL for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_LIVE_AGENT_BASE_URL
        return self.live_agent_base_url

    @property
    def resolved_openai_base_url(self) -> str | None:
        """Return the effective OpenAI-compatible base URL for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_OPENAI_BASE_URL
        return self.openai_base_url

    @property
    def resolved_openai_api_key_value(self) -> str | None:
        """Return the effective OpenAI-compatible API key for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_OPENAI_API_KEY
        if self.openai_api_key is None:
            return None
        return self.openai_api_key.get_secret_value()

    @property
    def resolved_openai_model(self) -> str:
        """Return the effective default chat model for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_OPENAI_MODEL
        return self.openai_model

    @property
    def resolved_planner_base_url(self) -> str | None:
        """Return the effective planner base URL for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_OPENAI_BASE_URL
        return self.planner_base_url or self.openai_base_url

    @property
    def resolved_planner_api_key_value(self) -> str | None:
        """Return the effective planner API key for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_OPENAI_API_KEY
        if self.planner_api_key is not None:
            planner_api_key = self.planner_api_key.get_secret_value().strip()
            if planner_api_key:
                return planner_api_key
        return self.resolved_openai_api_key_value

    @property
    def resolved_planner_model(self) -> str:
        """Return the effective planner model for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_OPENAI_MODEL
        return self.planner_model or self.resolved_openai_model

    @property
    def resolved_ragflow_base_url(self) -> str:
        """Return the effective RAGFlow base URL for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_RAGFLOW_BASE_URL
        return self.ragflow_base_url

    @property
    def resolved_ragflow_api_key_value(self) -> str | None:
        """Return the effective RAGFlow API key for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_RAGFLOW_API_KEY
        if self.ragflow_api_key is None:
            return None
        return self.ragflow_api_key.get_secret_value()

    @property
    def resolved_default_knowledge_dataset_id(self) -> str | None:
        """Return the effective default dataset id for the current environment."""

        if self.use_monitor_network_development_upstreams:
            return MONITOR_NETWORK_DEFAULT_KNOWLEDGE_DATASET_ID
        return self.default_knowledge_dataset_id

    @field_validator("openai_enable_thinking", "planner_enable_thinking", mode="before")
    @classmethod
    def _normalize_optional_boolean(cls, value: object) -> object:
        """将空字符串布尔配置视为未设置。"""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_rotate_when", mode="before")
    @classmethod
    def _normalize_log_rotate_when(cls, value: object) -> object:
        """统一日志轮转策略写法，兼容常见别名。"""

        if not isinstance(value, str):
            return value

        normalized_value = value.strip()
        if not normalized_value:
            return "midnight"

        aliases = {
            "daily": "midnight",
            "day": "midnight",
            "midnight": "midnight",
            "hourly": "H",
            "hour": "H",
            "minutes": "M",
            "minute": "M",
            "seconds": "S",
            "second": "S",
        }
        return aliases.get(normalized_value.lower(), normalized_value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取缓存后的应用配置。

    使用缓存是为了避免在一次请求链路中重复读取环境变量，
    同时也方便测试时显式清空缓存后重新装载配置。
    """

    return Settings()
