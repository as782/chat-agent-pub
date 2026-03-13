"""测试配置文件。

负责补齐测试运行时的项目根目录导入路径，并提供基础测试夹具。
当前阶段不负责复杂测试环境编排和外部依赖容器管理。
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """在每个测试前后清理配置缓存，避免环境变量互相污染。"""

    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def app_client() -> Iterator[TestClient]:
    """提供 FastAPI 测试客户端。"""

    from app.main import app

    with TestClient(app) as client:
        yield client
