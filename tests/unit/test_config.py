"""配置模块单元测试。"""

import pytest
from pytest import MonkeyPatch

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """确保每个测试都重新从环境变量加载配置，且不受仓库 .env 干扰。"""

    original_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    Settings.model_config["env_file"] = original_env_file


def test_get_settings_reads_environment_variables(monkeypatch: MonkeyPatch) -> None:
    """验证配置对象能够从环境变量读取值并正确组装数据库地址。"""

    monkeypatch.setenv("APP_NAME", "阶段二测试应用")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_TO_FILE", "true")
    monkeypatch.setenv("LOG_DIR", "runtime-logs")
    monkeypatch.setenv("LOG_FILE_NAME", "service.log")
    monkeypatch.setenv("LOG_ROTATE_WHEN", "daily")
    monkeypatch.setenv("LOG_ROTATE_INTERVAL", "2")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "9")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "65432")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "tester")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("OPENAI_MODEL", "test-chat-model")
    monkeypatch.setenv("OPENAI_ENABLE_THINKING", "true")
    monkeypatch.setenv("PLANNER_MODEL", "test-planner-model")
    monkeypatch.setenv("PLANNER_BASE_URL", "https://planner.example.com/v1")
    monkeypatch.setenv("PLANNER_API_KEY", "planner-test-key")
    monkeypatch.setenv("PLANNER_ENABLE_THINKING", "false")
    monkeypatch.setenv("PLANNER_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("DEFAULT_KNOWLEDGE_DATASET_ID", "dataset-default-001")
    monkeypatch.setenv("LIVE_AGENT_BASE_URL", "http://localhost:8081")
    monkeypatch.setenv("LIVE_AGENT_TIMEOUT_SECONDS", "18")
    monkeypatch.setenv("ENABLE_MONITOR_NETWORK_PROXY", "true")
    monkeypatch.setenv("RAGFLOW_TIMEOUT_SECONDS", "22")
    monkeypatch.setenv("MCP_HTTP_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("MCP_SSE_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("MCP_SSE_READ_TIMEOUT_SECONDS", "66")
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        '[{"name":"demo","transport":"http","endpoint":"https://mcp.example.com"}]',
    )

    settings = get_settings()

    assert settings.app_name == "阶段二测试应用"
    assert settings.app_env == "test"
    assert settings.enable_file_logging is True
    assert settings.log_dir == "runtime-logs"
    assert settings.log_file_name == "service.log"
    assert settings.log_rotate_when == "midnight"
    assert settings.log_rotate_interval == 2
    assert settings.log_backup_count == 9
    assert settings.database_url == "postgresql+asyncpg://tester:secret@localhost:65432/test_db"
    assert settings.redis_url.endswith("/1")
    assert settings.openai_base_url == "https://example.com/v1"
    assert settings.openai_timeout_seconds == 45.0
    assert settings.openai_model == "test-chat-model"
    assert settings.openai_enable_thinking is True
    assert settings.planner_model == "test-planner-model"
    assert settings.planner_base_url == "https://planner.example.com/v1"
    assert settings.planner_api_key is not None
    assert settings.planner_api_key.get_secret_value() == "planner-test-key"
    assert settings.planner_enable_thinking is False
    assert settings.planner_timeout_seconds == 12.0
    assert settings.default_knowledge_dataset_id == "dataset-default-001"
    assert settings.live_agent_base_url == "http://localhost:8081"
    assert settings.live_agent_timeout_seconds == 18.0
    assert settings.enable_monitor_network_proxy is True
    assert settings.ragflow_timeout_seconds == 22.0
    assert settings.mcp_http_timeout_seconds == 9.0
    assert settings.mcp_sse_timeout_seconds == 11.0
    assert settings.mcp_sse_read_timeout_seconds == 66.0
    assert settings.mcp_servers_json is not None
    assert settings.is_debug is True


def test_get_settings_prefers_explicit_postgres_dsn(monkeypatch: MonkeyPatch) -> None:
    """验证显式配置的数据库 DSN 优先级高于主机端口拼装。"""

    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "65432")
    monkeypatch.setenv("POSTGRES_DSN", "sqlite+aiosqlite:///override.db")

    settings = get_settings()

    assert settings.database_url == "sqlite+aiosqlite:///override.db"


def test_get_settings_treats_blank_optional_thinking_flags_as_none(
    monkeypatch: MonkeyPatch,
) -> None:
    """Blank optional thinking flags should be treated as unset."""

    monkeypatch.setenv("OPENAI_ENABLE_THINKING", "   ")
    monkeypatch.setenv("PLANNER_ENABLE_THINKING", "")

    settings = get_settings()

    assert settings.openai_enable_thinking is None
    assert settings.planner_enable_thinking is None


def test_get_settings_disables_file_logging_by_default_in_test_env(
    monkeypatch: MonkeyPatch,
) -> None:
    """验证 test 环境默认不会启用日志文件落盘。"""

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("LOG_TO_FILE", raising=False)

    settings = get_settings()

    assert settings.enable_file_logging is False
