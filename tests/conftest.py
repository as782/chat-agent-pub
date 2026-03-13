"""测试配置文件。

负责补齐测试运行时的项目根目录导入路径，并提供基础测试夹具。
当前阶段不负责复杂测试环境编排和外部依赖容器管理。
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """在每个测试前后清理配置缓存，避免环境变量互相污染。"""

    from app.core.config import get_settings
    from app.persistence.database import clear_database_caches

    get_settings.cache_clear()
    clear_database_caches()
    yield
    get_settings.cache_clear()
    clear_database_caches()


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[TestClient]:
    """提供使用临时 SQLite 数据库的 FastAPI 测试客户端。"""

    sqlite_database_path = tmp_path / "integration-test.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("POSTGRES_DSN", f"sqlite+aiosqlite:///{sqlite_database_path.as_posix()}")

    from app.main import create_app

    application = create_app()

    with TestClient(application) as client:
        yield client
