"""Agent 状态模型模块。
负责定义对话图执行时共享的状态结构、执行请求和计划模型。
当前阶段不负责长期记忆归档、多租户隔离和复杂任务编排持久化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from app.clients.llm_client import LlmInputMessage
from app.mcp.models import McpRuntimeTool
from app.schemas.knowledge import KnowledgeSearchResult
from app.tools.registry import ExecutedToolCall

AgentRoute = Literal["answer", "tool", "ragflow", "mcp", "traffic", "report"]
ProblemCategory = Literal[
    "policy",
    "route_planning",
    "traffic_status",
    "network_report",
    "general",
]
ExecutionMode = Literal["direct", "single_step", "multi_step"]
ExecutorType = Literal["answer", "rag", "mcp", "tool", "route", "traffic", "report"]


@dataclass(slots=True)
class ExecutionStep:
    """单个执行步骤定义。"""

    step_id: str
    executor: ExecutorType
    goal: str
    depends_on: list[str] = field(default_factory=list)
    can_run_in_parallel: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionPlan:
    """分类与执行计划结果。"""

    primary_category: ProblemCategory
    execution_mode: ExecutionMode
    recommended_route: AgentRoute
    need_clarification: bool = False
    clarification_question: str | None = None
    steps: list[ExecutionStep] = field(default_factory=list)


@dataclass(slots=True)
class ResolvedArguments:
    """当前问题的结构化参数提取结果。"""

    category: ProblemCategory
    arguments: dict[str, object] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    extraction_mode: str = "rule_based"


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
    enable_thinking: bool | None = None
    user_id: str | None = None
    message_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedContext:
    """对话图为当前轮次准备好的上下文。"""

    messages: list[LlmInputMessage]
    used_session_memory: bool
    memory_summary: str | None = None
    knowledge_context: str | None = None
    mcp_context: str | None = None
    traffic_context: str | None = None
    report_context: str | None = None
    answer_instruction: str | None = None


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
    enable_thinking: bool | None
    route: AgentRoute
    primary_category: ProblemCategory
    execution_plan: ExecutionPlan
    resolved_arguments: ResolvedArguments
    need_clarification: bool
    clarification_question: str | None
    knowledge_results: list[KnowledgeSearchResult]
    knowledge_context: str | None
    mcp_context: str | None
    traffic_context: str | None
    report_context: str | None
    mcp_tools: list[McpRuntimeTool]
    prepared_context: PreparedContext
    final_result: ChatTurnResult
    checkpoint_payload: dict[str, object] | None
