"""Agent 状态模型模块。
负责定义对话图执行时共享的状态结构、执行请求和计划模型。
当前阶段不负责长期记忆归档、多租户隔离和复杂任务编排持久化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage
from app.clients.llm_client import LlmInputMessage
from app.mcp.models import McpRuntimeTool
from app.schemas.knowledge import KnowledgeSearchResult
from app.tools.registry import ExecutedToolCall

AgentRoute = Literal["answer", "tool", "ragflow", "route", "mcp", "traffic", "service", "report"]
ProblemCategory = Literal[
    "policy",
    "route_planning",
    "traffic_status",
    "service_area",
    "network_report",
    "general",
]
ExecutionMode = Literal["direct", "single_step", "multi_step"]
ExecutorType = Literal["answer", "rag", "mcp", "tool", "route", "traffic", "service", "report"]


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
class ExecutorResult:
    """单个执行节点的标准化结果。"""

    step_id: str
    executor: ExecutorType
    is_success: bool
    raw_result: dict[str, object] = field(default_factory=dict)
    normalized_result: dict[str, object] = field(default_factory=dict)
    summary: str | None = None
    sources: list[str] = field(default_factory=list)
    error: str | None = None


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
    forced_route: AgentRoute | None = None
    scheduled_route: AgentRoute | None = None
    enable_thinking: bool | None = None
    user_id: str | None = None
    message_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedContext:
    """对话图为当前轮次准备好的上下文。"""

    messages: list[LlmInputMessage]
    used_session_memory: bool
    estimated_prompt_tokens: int | None = None
    memory_summary: str | None = None
    knowledge_context: str | None = None
    route_context: str | None = None
    mcp_context: str | None = None
    traffic_context: str | None = None
    service_context: str | None = None
    report_context: str | None = None
    answer_instruction: str | None = None
    executor_results_context: str | None = None


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
    route: str
    reasoning_content: str | None = None
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
    forced_route: AgentRoute | None
    route: AgentRoute
    scheduled_route: AgentRoute
    current_step_id: str | None
    runnable_step_ids: list[str]
    completed_step_ids: list[str]
    pending_step_ids: list[str]
    primary_category: ProblemCategory
    execution_plan: ExecutionPlan
    resolved_arguments: ResolvedArguments
    step_arguments: dict[str, ResolvedArguments]
    step_results: dict[str, ExecutorResult]
    need_clarification: bool
    clarification_question: str | None
    knowledge_results: list[KnowledgeSearchResult]
    knowledge_context: str | None
    route_context: str | None
    mcp_context: str | None
    traffic_context: str | None
    service_context: str | None
    report_context: str | None
    mcp_tools: list[McpRuntimeTool]
    prepared_context: PreparedContext
    tool_completion_result: AIMessage
    executed_tool_calls: list[ExecutedToolCall]
    final_result: ChatTurnResult
    checkpoint_payload: dict[str, object] | None


def resolve_execution_step_id(
    state: AgentState,
    *,
    executor: ExecutorType,
    default_step_id: str,
) -> str:
    """根据 execution_plan 为当前 executor 找到稳定的 step_id。"""

    execution_plan = state.get("execution_plan")
    if not isinstance(execution_plan, ExecutionPlan):
        return default_step_id

    for step in execution_plan.steps:
        if step.executor == executor:
            return step.step_id
    return default_step_id


def resolve_active_execution_step_id(
    state: AgentState,
    *,
    executor: ExecutorType,
    default_step_id: str,
) -> str:
    """优先返回当前调度中的 step_id，避免同类 executor 多步计划写回错误 step。"""

    current_step_id = state.get("current_step_id")
    if isinstance(current_step_id, str):
        current_step = get_execution_step(state, step_id=current_step_id)
        if current_step is not None and current_step.executor == executor:
            return current_step.step_id

    return resolve_execution_step_id(
        state,
        executor=executor,
        default_step_id=default_step_id,
    )


def get_execution_step(
    state: AgentState,
    *,
    step_id: str | None = None,
    executor: ExecutorType | None = None,
) -> ExecutionStep | None:
    """从 execution_plan 中按 step_id 或 executor 查找执行步骤。"""

    execution_plan = state.get("execution_plan")
    if not isinstance(execution_plan, ExecutionPlan):
        return None

    for step in execution_plan.steps:
        if step_id is not None and step.step_id == step_id:
            return step
        if executor is not None and step.executor == executor:
            return step
    return None


def merge_step_result(
    state: AgentState,
    *,
    result: ExecutorResult,
) -> dict[str, object]:
    """把当前节点结果合并进统一 step_results。"""

    step_results = dict(state.get("step_results", {}))
    existing_result = step_results.get(result.step_id)
    if not isinstance(existing_result, ExecutorResult):
        step_results[result.step_id] = result
        return {"step_results": step_results}

    step_results[result.step_id] = ExecutorResult(
        step_id=result.step_id,
        executor=existing_result.executor,
        is_success=existing_result.is_success and result.is_success,
        raw_result={**existing_result.raw_result, **result.raw_result},
        normalized_result={
            **existing_result.normalized_result,
            **result.normalized_result,
        },
        summary=result.summary or existing_result.summary,
        sources=list(dict.fromkeys([*existing_result.sources, *result.sources])),
        error=result.error or existing_result.error,
    )
    return {"step_results": step_results}


def resolve_step_arguments(
    state: AgentState,
    *,
    step_id: str | None = None,
    executor: ExecutorType | None = None,
) -> ResolvedArguments | None:
    """按 step_id 或 executor 读取当前步骤参数，兼容旧 resolved_arguments。"""

    step_arguments = state.get("step_arguments")
    if isinstance(step_arguments, dict):
        if step_id is not None:
            resolved_arguments = step_arguments.get(step_id)
            if isinstance(resolved_arguments, ResolvedArguments):
                return resolved_arguments
        if executor is not None:
            execution_step = get_execution_step(state, executor=executor)
            if execution_step is not None:
                resolved_arguments = step_arguments.get(execution_step.step_id)
                if isinstance(resolved_arguments, ResolvedArguments):
                    return resolved_arguments

    fallback_arguments = state.get("resolved_arguments")
    if isinstance(fallback_arguments, ResolvedArguments):
        return fallback_arguments
    return None
