"""上下文构建单元测试。"""

from __future__ import annotations

import pytest

import app.agent.context_builder as context_builder_module
from app.agent.context_builder import ContextBuilder
from app.agent.prompts import MEMORY_SUMMARY_PROMPT_PREFIX
from app.clients.llm_client import LlmInputMessage

FIXED_CURRENT_DATETIME_CONTEXT = (
    "以下是当前系统时间信息，仅用于时间判断和日期换算：\n"
    "当前时区: Asia/Shanghai\n"
    "当前时间: 2026-04-10T14:00:00+08:00"
)


@pytest.fixture(autouse=True)
def _patch_current_datetime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """固定当前时间上下文，避免单测依赖真实时钟。"""

    monkeypatch.setattr(
        context_builder_module,
        "_build_current_datetime_context",
        lambda timezone_name="Asia/Shanghai": FIXED_CURRENT_DATETIME_CONTEXT,
    )


def test_context_builder_uses_only_input_messages_without_session_memory() -> None:
    """验证未启用会话记忆时，只使用本次请求显式传入的消息。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[
            LlmInputMessage(role="system", content="如果不知道就说不知道"),
            LlmInputMessage(role="user", content="我刚刚告诉你的名字是什么？"),
        ],
        recent_messages=[
            LlmInputMessage(role="user", content="我叫小王"),
            LlmInputMessage(role="assistant", content="好的，我记住了"),
        ],
        memory_summary="用户曾提到自己叫小王。",
        need_session_memory=False,
    )

    assert prepared_context.used_session_memory is False
    assert [message.role for message in prepared_context.messages] == [
        "system",
        "system",
        "user",
    ]
    assert prepared_context.messages[0].content == FIXED_CURRENT_DATETIME_CONTEXT
    assert prepared_context.messages[1].content == "如果不知道就说不知道"
    assert prepared_context.memory_summary is None


def test_context_builder_injects_summary_and_recent_history_for_single_user_message() -> None:
    """验证启用会话记忆后，会自动注入摘要与最近历史。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[LlmInputMessage(role="user", content="我刚刚告诉你的名字是什么？")],
        recent_messages=[
            LlmInputMessage(role="user", content="我叫小王"),
            LlmInputMessage(role="assistant", content="好的，我记住了"),
            LlmInputMessage(role="user", content="我刚刚告诉你的名字是什么？"),
        ],
        memory_summary="用户此前自称叫小王。",
        need_session_memory=True,
    )

    assert prepared_context.used_session_memory is True
    assert prepared_context.messages[0].role == "system"
    assert prepared_context.messages[0].content == FIXED_CURRENT_DATETIME_CONTEXT
    assert "用户此前自称叫小王" in prepared_context.messages[1].content
    assert prepared_context.messages[-1].content == "我刚刚告诉你的名字是什么？"
    assert [message.content for message in prepared_context.messages].count(
        "我刚刚告诉你的名字是什么？"
    ) == 1


def test_context_builder_merges_session_history_and_explicit_messages_when_session_exists() -> None:
    """验证启用会话记忆后，会合并系统历史和本次请求消息。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[
            LlmInputMessage(role="system", content="请同时参考系统记录和本次输入。"),
            LlmInputMessage(role="user", content="我刚刚告诉你的名字是什么？"),
        ],
        recent_messages=[
            LlmInputMessage(role="user", content="我叫小王"),
            LlmInputMessage(role="assistant", content="好的，我记住了"),
            LlmInputMessage(role="user", content="我刚刚告诉你的名字是什么？"),
        ],
        memory_summary="用户此前自称叫小王。",
        need_session_memory=True,
    )

    assert prepared_context.used_session_memory is True
    assert prepared_context.messages[0].role == "system"
    assert prepared_context.messages[0].content == FIXED_CURRENT_DATETIME_CONTEXT
    assert "用户此前自称叫小王" in prepared_context.messages[1].content
    assert [message.content for message in prepared_context.messages] == [
        FIXED_CURRENT_DATETIME_CONTEXT,
        f"{MEMORY_SUMMARY_PROMPT_PREFIX}用户此前自称叫小王。",
        "请同时参考系统记录和本次输入。",
        "我叫小王",
        "好的，我记住了",
        "我刚刚告诉你的名字是什么？",
    ]


def test_context_builder_moves_explicit_system_messages_before_history_when_session_exists(
) -> None:
    """验证带会话记忆时，显式 system 消息会被提升到所有非 system 消息之前。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[
            LlmInputMessage(role="user", content="我刚刚告诉你的名字是什么？"),
            LlmInputMessage(role="system", content="如果不知道就说不知道。"),
        ],
        recent_messages=[
            LlmInputMessage(role="user", content="我叫小王"),
            LlmInputMessage(role="assistant", content="好的，我记住了"),
        ],
        memory_summary="用户此前自称叫小王。",
        need_session_memory=True,
    )

    assert [message.role for message in prepared_context.messages[:2]] == ["system", "system"]
    assert prepared_context.messages[0].content == FIXED_CURRENT_DATETIME_CONTEXT
    assert prepared_context.messages[1].content == f"{MEMORY_SUMMARY_PROMPT_PREFIX}用户此前自称叫小王。"
    assert prepared_context.messages[2].content == "如果不知道就说不知道。"
    assert [message.role for message in prepared_context.messages[3:]] == [
        "user",
        "assistant",
        "user",
    ]


def test_context_builder_moves_explicit_system_messages_to_front_without_session_memory() -> None:
    """验证未启用会话记忆时，也会把散落的 system 消息统一提升到前部。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[
            LlmInputMessage(role="user", content="你好"),
            LlmInputMessage(role="system", content="请简洁回答。"),
        ],
        recent_messages=[],
        memory_summary=None,
        need_session_memory=False,
    )

    assert [message.role for message in prepared_context.messages] == [
        "system",
        "system",
        "user",
    ]
    assert [message.content for message in prepared_context.messages] == [
        FIXED_CURRENT_DATETIME_CONTEXT,
        "请简洁回答。",
        "你好",
    ]


def test_context_builder_includes_knowledge_context_as_system_message() -> None:
    """验证知识库上下文会以 system 消息的方式注入模型输入。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[LlmInputMessage(role="user", content="知识库: 西湖在哪里？")],
        recent_messages=[],
        memory_summary=None,
        need_session_memory=False,
        knowledge_context="以下是知识库检索结果：西湖位于杭州。",
    )

    assert prepared_context.knowledge_context == "以下是知识库检索结果：西湖位于杭州。"
    assert [message.role for message in prepared_context.messages] == [
        "system",
        "system",
        "user",
    ]
    assert prepared_context.messages[0].content == FIXED_CURRENT_DATETIME_CONTEXT
    assert prepared_context.messages[1].content == "以下是知识库检索结果：西湖位于杭州。"


def test_context_builder_includes_mcp_context_as_system_message() -> None:
    """验证 MCP 上下文会以 system 消息的方式注入模型输入。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[LlmInputMessage(role="user", content="mcp: 当前有哪些服务？")],
        recent_messages=[],
        memory_summary=None,
        need_session_memory=False,
        mcp_context="以下是当前系统已配置的 MCP 服务骨架信息。",
    )

    assert prepared_context.mcp_context == "以下是当前系统已配置的 MCP 服务骨架信息。"
    assert [message.role for message in prepared_context.messages] == [
        "system",
        "system",
        "user",
    ]
    assert prepared_context.messages[0].content == FIXED_CURRENT_DATETIME_CONTEXT
    assert prepared_context.messages[1].content == "以下是当前系统已配置的 MCP 服务骨架信息。"


def test_context_builder_includes_answer_instruction_and_business_contexts() -> None:
    """验证分类总结提示词和业务上下文会作为 system 消息注入。"""

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[LlmInputMessage(role="user", content="请输出今天全路网路况表格")],
        recent_messages=[],
        memory_summary=None,
        need_session_memory=False,
        answer_instruction="请优先输出表格，再给出总结。",
        executor_results_context="以下是执行节点返回的结构化结果：{report_count: 1}",
        route_context="以下是路线规划查询参数：{origin: 杭州, destination: 金华}",
        traffic_context="以下是路况查询参数：{target: 杭金衢高速}",
        service_context="以下是服务区查询参数：{keyword: 杭州东服务区}",
        report_context="以下是路网报告任务参数：{scope: 全路网}",
    )

    assert prepared_context.answer_instruction == "请优先输出表格，再给出总结。"
    assert (
        prepared_context.executor_results_context
        == "以下是执行节点返回的结构化结果：{report_count: 1}"
    )
    assert (
        prepared_context.route_context
        == "以下是路线规划查询参数：{origin: 杭州, destination: 金华}"
    )
    assert prepared_context.traffic_context == "以下是路况查询参数：{target: 杭金衢高速}"
    assert prepared_context.service_context == "以下是服务区查询参数：{keyword: 杭州东服务区}"
    assert prepared_context.report_context == "以下是路网报告任务参数：{scope: 全路网}"
    assert [message.role for message in prepared_context.messages] == [
        "system",
        "system",
        "system",
        "system",
        "system",
        "system",
        "system",
        "user",
    ]


def test_context_builder_exposes_estimated_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """楠岃瘉鍑嗗鍚庣殑涓婁笅鏂囦細鎶婇浼拌 token 鏁版帴鍑恒€?"""

    monkeypatch.setattr(
        context_builder_module,
        "estimate_messages_tokens",
        lambda messages, model_name=None: 123,
    )

    builder = ContextBuilder()
    prepared_context = builder.build_context(
        input_messages=[LlmInputMessage(role="user", content="你好")],
        recent_messages=[],
        memory_summary=None,
        need_session_memory=False,
        model_name="gpt-4o-mini",
    )

    assert prepared_context.estimated_prompt_tokens == 123
