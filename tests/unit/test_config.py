"""配置模块单元测试。"""

from pytest import MonkeyPatch

from app.core.config import get_settings


def test_get_settings_reads_environment_variables(monkeypatch: MonkeyPatch) -> None:
    """验证配置对象能够从环境变量读取值。"""

    monkeypatch.setenv("APP_NAME", "阶段二测试应用")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+asyncpg://user:pass@localhost:5432/test_db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")

    settings = get_settings()

    assert settings.app_name == "阶段二测试应用"
    assert settings.app_env == "test"
    assert settings.postgres_dsn.endswith("/test_db")
    assert settings.redis_url.endswith("/1")
    assert settings.is_debug is True
