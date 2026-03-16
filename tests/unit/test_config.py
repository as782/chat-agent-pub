"""配置模块单元测试。"""

from pytest import MonkeyPatch

from app.core.config import get_settings


def test_get_settings_reads_environment_variables(monkeypatch: MonkeyPatch) -> None:
    """验证配置对象能够从环境变量读取值并正确组装数据库地址。"""

    monkeypatch.setenv("APP_NAME", "阶段二测试应用")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "65432")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "tester")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-chat-model")

    settings = get_settings()

    assert settings.app_name == "阶段二测试应用"
    assert settings.app_env == "test"
    assert settings.database_url == "postgresql+asyncpg://tester:secret@localhost:65432/test_db"
    assert settings.redis_url.endswith("/1")
    assert settings.openai_base_url == "https://example.com/v1"
    assert settings.openai_model == "test-chat-model"
    assert settings.is_debug is True


def test_get_settings_prefers_explicit_postgres_dsn(monkeypatch: MonkeyPatch) -> None:
    """验证显式配置的数据库 DSN 优先级高于主机端口拼装。"""

    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "65432")
    monkeypatch.setenv("POSTGRES_DSN", "sqlite+aiosqlite:///override.db")

    settings = get_settings()

    assert settings.database_url == "sqlite+aiosqlite:///override.db"
