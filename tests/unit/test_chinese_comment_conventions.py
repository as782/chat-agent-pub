"""中文注释规范测试模块。
负责对核心模块文件说明和关键函数中文 docstring 做基线校验。
当前阶段仅做自动化最小检查，不负责替代人工代码评审。
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

CHINESE_TEXT_PATTERN = re.compile(r"[\u4e00-\u9fff]")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORE_MODULE_FILES = [
    "app/main.py",
    "app/core/config.py",
    "app/core/logger.py",
    "app/core/exceptions.py",
    "app/agent/graph.py",
    "app/agent/context_builder.py",
    "app/memory/manager.py",
    "app/knowledge/service.py",
    "app/knowledge/ragflow/client.py",
    "app/tools/registry.py",
    "app/tools/builtin/calculator.py",
    "app/mcp/manager.py",
    "app/persistence/session_repo.py",
]

REQUIRED_DOCSTRING_TARGETS = [
    ("app.api.v1.chat", None, "create_chat_completion"),
    ("app.api.v1.sessions", None, "create_session"),
    ("app.api.v1.messages", None, "list_session_messages"),
    ("app.api.v1.knowledge", None, "retrieve_knowledge"),
    ("app.api.v1.mcp", None, "list_mcp_servers"),
    ("app.services.chat_service", "ChatService", "send_message"),
    ("app.services.session_service", "SessionService", "create_session"),
    ("app.services.message_service", "MessageService", "list_messages"),
    ("app.knowledge.service", "KnowledgeService", "sync_datasets"),
    ("app.persistence.session_repo", "SessionRepository", "create"),
    ("app.persistence.message_repo", "MessageRepository", "list_by_session"),
    ("app.persistence.memory_repo", "MemoryRepository", "upsert"),
    ("app.persistence.ragflow_repo", "RagflowRepository", "upsert_dataset"),
    ("app.agent.nodes.router_node", "RouterNode", "run"),
    ("app.agent.nodes.answer_node", "AnswerNode", "run"),
    ("app.agent.nodes.memory_node", "MemoryNode", "run"),
    ("app.agent.nodes.ragflow_node", "RagflowNode", "run"),
    ("app.agent.nodes.tool_node", "ToolNode", "run"),
    ("app.agent.nodes.mcp_node", "McpNode", "run"),
    ("app.knowledge.ragflow.client", "RagflowClient", "request"),
    ("app.tools.builtin.calculator", None, "calculator_tool"),
    ("app.tools.builtin.datetime_tool", None, "current_datetime_tool"),
    ("app.agent.context_builder", "ContextBuilder", "build_context"),
]


def test_core_modules_have_chinese_module_docstrings() -> None:
    """验证核心模块文件开头都带中文文件说明。"""

    for relative_path in CORE_MODULE_FILES:
        source_text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        module = ast.parse(source_text)
        module_docstring = ast.get_docstring(module)

        assert module_docstring, f"{relative_path} 缺少模块级 docstring"
        assert CHINESE_TEXT_PATTERN.search(module_docstring), (
            f"{relative_path} 的模块级 docstring 需要包含中文说明"
        )


def test_key_callables_have_chinese_docstrings() -> None:
    """验证关键 API、service、repository、节点和工具函数带中文 docstring。"""

    for module_name, owner_name, callable_name in REQUIRED_DOCSTRING_TARGETS:
        module = importlib.import_module(module_name)
        owner = getattr(module, owner_name) if owner_name is not None else module
        target_callable = getattr(owner, callable_name)
        resolved_callable = (
            target_callable.func if hasattr(target_callable, "func") else target_callable
        )
        docstring = inspect.getdoc(inspect.unwrap(resolved_callable))

        assert docstring, f"{module_name}.{callable_name} 缺少 docstring"
        assert CHINESE_TEXT_PATTERN.search(docstring), (
            f"{module_name}.{callable_name} 的 docstring 需要包含中文说明"
        )
