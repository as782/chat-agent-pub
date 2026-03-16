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

from app.clients.llm_client import LlmChatCompletionResult

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
    """提供使用临时 SQLite 数据库和假 LLM 的 FastAPI 测试客户端。"""

    sqlite_database_path = tmp_path / "integration-test.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("POSTGRES_DSN", f"sqlite+aiosqlite:///{sqlite_database_path.as_posix()}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    async def fake_create_chat_completion(
        self: object,
        messages: list[tuple[str, str]],
        model_name: str | None = None,
    ) -> LlmChatCompletionResult:
        """为集成测试返回稳定的假模型回答与元数据。"""

        latest_user_message = ""
        for role, content in reversed(messages):
            if role == "user":
                latest_user_message = content
                break

        return LlmChatCompletionResult(
            content=f"测试模型回答：{latest_user_message}",
            model_name=model_name or "test-model",
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )

    from app.main import create_app

    application = create_app()

    with TestClient(application) as client:
        yield client
