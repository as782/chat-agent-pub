"""Agent 状态模型模块。
负责定义对话图执行时共享的状态结构、执行请求和图输出结果。
当前阶段不负责跨会话的长期记忆归档和多租户隔离。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from app.clients.llm_client import LlmInputMessage
from app.tools.registry import ExecutedToolCall

AgentRoute = Literal["answer", "tool", "ragflow", "mcp"]


@dataclass(slots=True)
class ChatExecutionRequest:
    """内部聊天执行请求。"""

    session_id: str | None
    need_session_memory: bool
    latest_user_message: str
    input_messages: list[LlmInputMessage]
    model_name: str | None
    requested_tool_names: list[str] | None
    tool_choice: str | dict[str, object] | None
    user_id: str | None = None
    message_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedContext:
    """对话图为当前轮次准备好的上下文。"""

    messages: list[LlmInputMessage]
    used_session_memory: bool
    memory_summary: str | None = None


@dataclass(slots=True)
class ChatTurnResult:
    """单轮对话执行结果。"""

    session_id: str
    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)
    used_session_memory: bool = False


class AgentState(TypedDict, total=False):
    """LangGraph 执行状态。"""

    session_id: str
    need_session_memory: bool
    user_id: str | None
    latest_user_message: str
    input_messages: list[LlmInputMessage]
    model_name: str | None
    requested_tool_names: list[str] | None
    tool_choice: str | dict[str, object] | None
    route: AgentRoute
    prepared_context: PreparedContext
    final_result: ChatTurnResult
    checkpoint_payload: dict[str, object] | None
