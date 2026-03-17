"""上下文构建单元测试。"""

from app.agent.context_builder import ContextBuilder
from app.clients.llm_client import LlmInputMessage


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
    assert [message.role for message in prepared_context.messages] == ["system", "user"]
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
    assert "用户此前自称叫小王" in prepared_context.messages[0].content
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
    assert "用户此前自称叫小王" in prepared_context.messages[0].content
    assert [message.content for message in prepared_context.messages] == [
        "以下是当前会话的历史摘要，仅在不与用户本次显式输入冲突时参考：\n用户此前自称叫小王。",
        "我叫小王",
        "好的，我记住了",
        "请同时参考系统记录和本次输入。",
        "我刚刚告诉你的名字是什么？",
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
    assert [message.role for message in prepared_context.messages] == ["system", "user"]
    assert prepared_context.messages[0].content == "以下是知识库检索结果：西湖位于杭州。"


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
    assert [message.role for message in prepared_context.messages] == ["system", "user"]
    assert prepared_context.messages[0].content == "以下是当前系统已配置的 MCP 服务骨架信息。"


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
        traffic_context="以下是路况查询参数：{target: 杭金衢高速}",
        report_context="以下是路网报告任务参数：{scope: 全路网}",
    )

    assert prepared_context.answer_instruction == "请优先输出表格，再给出总结。"
    assert (
        prepared_context.executor_results_context
        == "以下是执行节点返回的结构化结果：{report_count: 1}"
    )
    assert prepared_context.traffic_context == "以下是路况查询参数：{target: 杭金衢高速}"
    assert prepared_context.report_context == "以下是路网报告任务参数：{scope: 全路网}"
    assert [message.role for message in prepared_context.messages] == [
        "system",
        "system",
        "system",
        "system",
        "user",
    ]
