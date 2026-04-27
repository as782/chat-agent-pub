"""上下文构建工具。

负责组装最终发送给模型的消息列表，并提供安全的 token 估算与上下文截断能力。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from json import dumps
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import tiktoken

from app.agent.history_utils import limit_messages_to_recent_turns
from app.agent.prompts import (
    CURRENT_DATETIME_CONTEXT_PROMPT_PREFIX,
    MEMORY_SUMMARY_PROMPT_PREFIX,
)
from app.agent.state import PreparedContext
from app.clients.llm_client import LlmInputMessage, LlmToolCall
from app.persistence.models import MessageEntity

# 当前本地 qwen3535ba3b 的 SGLang 服务虽然配置了 262144 上下文窗口，
# 但业务侧保守控制在 80k 内，给输出、工具轮次、估算误差和并发显存留余量。
MAX_PROMPT_TOKENS = 80_000

# 路线规划上下文包含路线、服务区、管制、收费站和拥堵明细，高峰期容易膨胀。
MAX_ROUTE_CONTEXT_TOKENS = 25_000

# 路况上下文可能按多条道路展开，每条道路又包含拥堵、管制、收费站等事件。
MAX_TRAFFIC_CONTEXT_TOKENS = 25_000

# 普通意图被分类/兜底进入 report 时，先使用较保守的 report_content 预算。
MAX_REPORT_CONTEXT_TOKENS = 30_000

# 明确 scheduled_route=report 时，用户通常需要全网/全省报表，允许更完整的上下文。
MAX_SCHEDULED_REPORT_CONTEXT_TOKENS = 60_000

# 工具或 MCP 返回可能是大 JSON；原始结果可持久化，进入模型前必须压缩。
MAX_TOOL_OUTPUT_TOKENS = 20_000

# 执行节点结果只保留组织最终回答所需的 compact 信息，避免和专用上下文重复。
MAX_EXECUTOR_RESULTS_TOKENS = 8_000

# MCP、知识库、服务区、会话记忆属于辅助上下文，避免挤占 route/traffic/report 预算。
MAX_MCP_CONTEXT_TOKENS = 10_000
MAX_KNOWLEDGE_CONTEXT_TOKENS = 10_000
MAX_SERVICE_CONTEXT_TOKENS = 10_000
MAX_MEMORY_SUMMARY_TOKENS = 8_000

# 整体 prompt 兜底压缩时，每条被截断的非 user 消息至少保留这部分上下文。
MIN_TRUNCATED_MESSAGE_TOKENS = 512

_TRUNCATION_NOTICE_TEMPLATE = (
    "\n\n[系统提示：{label} 数据量较大，后续明细已省略；"
    "请基于已保留的统计、重点异常和代表性明细回答，不要编造被省略的数据。]"
)


def message_entity_to_input_message(message_entity: MessageEntity) -> LlmInputMessage:
    """Convert a persisted message entity to an input message."""

    message_metadata = message_entity.message_metadata or {}
    tool_calls_payload = message_metadata.get("tool_calls", [])
    parsed_tool_calls: list[LlmToolCall] = []

    if isinstance(tool_calls_payload, list):
        for tool_call_payload in tool_calls_payload:
            if not isinstance(tool_call_payload, dict):
                continue
            tool_arguments = tool_call_payload.get("arguments", {})
            parsed_tool_calls.append(
                LlmToolCall(
                    tool_call_id=str(tool_call_payload.get("tool_call_id", "")),
                    tool_name=str(tool_call_payload.get("tool_name", "")),
                    arguments=tool_arguments if isinstance(tool_arguments, dict) else {},
                )
            )

    tool_call_id = message_metadata.get("tool_call_id")
    return LlmInputMessage(
        role=message_entity.role,
        content=message_entity.content,
        tool_call_id=str(tool_call_id) if tool_call_id is not None else None,
        tool_calls=parsed_tool_calls,
    )


def serialize_input_messages(messages: Sequence[LlmInputMessage]) -> list[dict[str, Any]]:
    """Serialize messages for memory storage and checkpoints."""

    return [
        {
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "tool_calls": [
                {
                    "tool_call_id": tool_call.tool_call_id,
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in message.tool_calls
            ],
        }
        for message in messages
    ]


def deserialize_input_messages(message_payloads: Sequence[dict[str, Any]]) -> list[LlmInputMessage]:
    """Restore serialized messages from checkpoint or memory storage."""

    deserialized_messages: list[LlmInputMessage] = []
    for message_payload in message_payloads:
        tool_calls_payload = message_payload.get("tool_calls", [])
        parsed_tool_calls: list[LlmToolCall] = []
        if isinstance(tool_calls_payload, list):
            for tool_call_payload in tool_calls_payload:
                if not isinstance(tool_call_payload, dict):
                    continue
                tool_arguments = tool_call_payload.get("arguments", {})
                parsed_tool_calls.append(
                    LlmToolCall(
                        tool_call_id=str(tool_call_payload.get("tool_call_id", "")),
                        tool_name=str(tool_call_payload.get("tool_name", "")),
                        arguments=tool_arguments if isinstance(tool_arguments, dict) else {},
                    )
                )

        deserialized_messages.append(
            LlmInputMessage(
                role=str(message_payload.get("role", "user")),
                content=str(message_payload.get("content", "")),
                name=(
                    str(message_payload["name"])
                    if message_payload.get("name") is not None
                    else None
                ),
                tool_call_id=(
                    str(message_payload["tool_call_id"])
                    if message_payload.get("tool_call_id") is not None
                    else None
                ),
                tool_calls=parsed_tool_calls,
            )
        )

    return deserialized_messages


def _build_current_datetime_context(timezone_name: str = "Asia/Shanghai") -> str:
    """Build a system prompt with the current timezone and time."""

    normalized_timezone_name = timezone_name.strip() or "Asia/Shanghai"
    try:
        timezone = ZoneInfo(normalized_timezone_name)
    except ZoneInfoNotFoundError:
        timezone = UTC
        normalized_timezone_name = "UTC"

    current_time = datetime.now(timezone)
    return (
        f"{CURRENT_DATETIME_CONTEXT_PROMPT_PREFIX}"
        f"当前时区: {normalized_timezone_name}\n"
        f"当前时间: {current_time.isoformat(timespec='seconds')}"
    )


def _get_token_encoding(model_name: str) -> tiktoken.Encoding:
    """Resolve a tokenizer for the requested model name, with safe fallbacks."""

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        raise


_TEXT_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def _estimate_text_tokens_locally(text: str) -> int:
    """Estimate tokens without relying on tokenizer downloads or network access."""

    normalized_text = text.strip()
    if not normalized_text:
        return 0

    token_count = 0
    for piece in _TEXT_TOKEN_PATTERN.findall(normalized_text):
        if piece.isascii() and (piece.isalnum() or "_" in piece):
            token_count += max(1, math.ceil(len(piece) / 4))
        else:
            token_count += 1
    return token_count


def _build_truncation_notice(label: str) -> str:
    return _TRUNCATION_NOTICE_TEMPLATE.format(label=label)


def truncate_text_to_token_budget(
    text: str | None,
    *,
    max_tokens: int,
    label: str,
) -> str | None:
    """按近似 token 预算截断大块上下文文本。"""

    if text is None:
        return None

    normalized_text = text.strip()
    if not normalized_text:
        return normalized_text

    if _estimate_text_tokens_locally(normalized_text) <= max_tokens:
        return normalized_text

    notice = _build_truncation_notice(label)
    notice_tokens = _estimate_text_tokens_locally(notice)
    content_budget = max(0, max_tokens - notice_tokens)
    if content_budget <= 0:
        return notice.strip()

    low = 0
    high = len(normalized_text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = normalized_text[:mid].rstrip()
        if _estimate_text_tokens_locally(candidate) <= content_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    return f"{best}{notice}"


def truncate_tool_output_for_context(output: str) -> str:
    """在工具或 MCP 输出进入下一轮模型前做截断保护。"""

    return truncate_text_to_token_budget(
        output,
        max_tokens=MAX_TOOL_OUTPUT_TOKENS,
        label="工具返回结果",
    ) or ""


def _estimate_messages_tokens_locally(messages: Sequence[LlmInputMessage]) -> int:
    """Fallback estimator used when tokenizer resolution is unavailable."""

    total_tokens = 3
    for message in messages:
        total_tokens += 3
        total_tokens += _estimate_text_tokens_locally(message.role)
        total_tokens += _estimate_text_tokens_locally(message.content)
        if message.name:
            total_tokens += 1 + _estimate_text_tokens_locally(message.name)
        if message.tool_call_id:
            total_tokens += 1 + _estimate_text_tokens_locally(message.tool_call_id)
        if message.tool_calls:
            tool_calls_payload = [
                {
                    "tool_call_id": tool_call.tool_call_id,
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in message.tool_calls
            ]
            total_tokens += _estimate_text_tokens_locally(
                dumps(tool_calls_payload, ensure_ascii=False, separators=(",", ":"))
            )
    return total_tokens


def estimate_messages_tokens(
    messages: Sequence[LlmInputMessage],
    *,
    model_name: str | None = None,
) -> int:
    """Estimate the token count for a list of input messages.

    This is intentionally approximate and safe for custom model names.
    """

    normalized_model_name = (model_name or "gpt-4o-mini").strip() or "gpt-4o-mini"
    try:
        encoding = _get_token_encoding(normalized_model_name)
    except Exception:
        return _estimate_messages_tokens_locally(messages)

    # Approximate ChatML-style overhead: a few tokens per message plus the
    # encoded content. We prefer stability over exact parity with a provider.
    total_tokens = 3
    try:
        for message in messages:
            total_tokens += 3
            total_tokens += len(encoding.encode(message.role))
            total_tokens += len(encoding.encode(message.content.strip()))
            if message.name:
                total_tokens += 1 + len(encoding.encode(message.name))
            if message.tool_call_id:
                total_tokens += 1 + len(encoding.encode(message.tool_call_id))
            if message.tool_calls:
                tool_calls_payload = [
                    {
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in message.tool_calls
                ]
                total_tokens += len(
                    encoding.encode(
                        dumps(tool_calls_payload, ensure_ascii=False, separators=(",", ":"))
                    )
                )
        return total_tokens
    except Exception:
        return _estimate_messages_tokens_locally(messages)


class ContextBuilder:
    """Build the final prompt context for the current turn."""

    @staticmethod
    def _split_system_messages(
        messages: Sequence[LlmInputMessage],
    ) -> tuple[list[LlmInputMessage], list[LlmInputMessage]]:
        """Split system and non-system messages."""

        system_messages = [message for message in messages if message.role == "system"]
        non_system_messages = [message for message in messages if message.role != "system"]
        return system_messages, non_system_messages

    @staticmethod
    def _build_internal_system_messages(
        *,
        answer_instruction: str | None,
        current_datetime_context: str | None,
        executor_results_context: str | None,
        memory_summary: str | None,
        knowledge_context: str | None,
        route_context: str | None,
        mcp_context: str | None,
        traffic_context: str | None,
        service_context: str | None,
        report_context: str | None,
    ) -> list[LlmInputMessage]:
        """Construct internally injected system messages."""

        system_messages: list[LlmInputMessage] = []
        if answer_instruction:
            system_messages.append(LlmInputMessage(role="system", content=answer_instruction))
        if current_datetime_context:
            system_messages.append(LlmInputMessage(role="system", content=current_datetime_context))
        if executor_results_context:
            system_messages.append(
                LlmInputMessage(role="system", content=executor_results_context)
            )
        if memory_summary:
            system_messages.append(
                LlmInputMessage(
                    role="system",
                    content=f"{MEMORY_SUMMARY_PROMPT_PREFIX}{memory_summary}",
                )
            )
        if knowledge_context:
            system_messages.append(LlmInputMessage(role="system", content=knowledge_context))
        if route_context:
            system_messages.append(LlmInputMessage(role="system", content=route_context))
        if mcp_context:
            system_messages.append(LlmInputMessage(role="system", content=mcp_context))
        if traffic_context:
            system_messages.append(LlmInputMessage(role="system", content=traffic_context))
        if service_context:
            system_messages.append(LlmInputMessage(role="system", content=service_context))
        if report_context:
            system_messages.append(LlmInputMessage(role="system", content=report_context))
        return system_messages

    @staticmethod
    def _enforce_prompt_token_budget(
        messages: Sequence[LlmInputMessage],
        *,
        model_name: str | None,
        max_tokens: int = MAX_PROMPT_TOKENS,
    ) -> list[LlmInputMessage]:
        """在分块截断之后，对整个 prompt 再做最后一层预算保护。"""

        fitted_messages = list(messages)
        current_tokens = estimate_messages_tokens(fitted_messages, model_name=model_name)
        if current_tokens <= max_tokens:
            return fitted_messages

        for _ in range(len(fitted_messages) * 2):
            current_tokens = estimate_messages_tokens(fitted_messages, model_name=model_name)
            if current_tokens <= max_tokens:
                break

            candidates: list[tuple[int, int]] = []
            for index, message in enumerate(fitted_messages):
                if message.role == "user":
                    continue
                message_tokens = _estimate_text_tokens_locally(message.content)
                if message_tokens > MIN_TRUNCATED_MESSAGE_TOKENS:
                    candidates.append((message_tokens, index))

            if not candidates:
                break

            message_tokens, index = max(candidates)
            overflow_tokens = current_tokens - max_tokens
            next_budget = max(
                MIN_TRUNCATED_MESSAGE_TOKENS,
                message_tokens - overflow_tokens - 1_000,
            )
            message = fitted_messages[index]
            fitted_messages[index] = LlmInputMessage(
                role=message.role,
                content=truncate_text_to_token_budget(
                    message.content,
                    max_tokens=next_budget,
                    label="整体上下文",
                )
                or "",
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_calls=message.tool_calls,
            )

        return fitted_messages

    def build_context(
        self,
        *,
        input_messages: Sequence[LlmInputMessage],
        recent_messages: Sequence[LlmInputMessage],
        memory_summary: str | None,
        need_session_memory: bool,
        max_turns: int = 1,
        model_name: str | None = None,
        answer_instruction: str | None = None,
        executor_results_context: str | None = None,
        knowledge_context: str | None = None,
        route_context: str | None = None,
        mcp_context: str | None = None,
        traffic_context: str | None = None,
        service_context: str | None = None,
        report_context: str | None = None,
        report_context_max_tokens: int = MAX_REPORT_CONTEXT_TOKENS,
    ) -> PreparedContext:
        """Build the final model input for the current turn."""

        executor_results_context = truncate_text_to_token_budget(
            executor_results_context,
            max_tokens=MAX_EXECUTOR_RESULTS_TOKENS,
            label="执行节点结果",
        )
        memory_summary = truncate_text_to_token_budget(
            memory_summary,
            max_tokens=MAX_MEMORY_SUMMARY_TOKENS,
            label="会话记忆",
        )
        knowledge_context = truncate_text_to_token_budget(
            knowledge_context,
            max_tokens=MAX_KNOWLEDGE_CONTEXT_TOKENS,
            label="知识库上下文",
        )
        route_context = truncate_text_to_token_budget(
            route_context,
            max_tokens=MAX_ROUTE_CONTEXT_TOKENS,
            label="路线规划上下文",
        )
        mcp_context = truncate_text_to_token_budget(
            mcp_context,
            max_tokens=MAX_MCP_CONTEXT_TOKENS,
            label="MCP 上下文",
        )
        traffic_context = truncate_text_to_token_budget(
            traffic_context,
            max_tokens=MAX_TRAFFIC_CONTEXT_TOKENS,
            label="路况上下文",
        )
        service_context = truncate_text_to_token_budget(
            service_context,
            max_tokens=MAX_SERVICE_CONTEXT_TOKENS,
            label="服务区上下文",
        )
        report_context = truncate_text_to_token_budget(
            report_context,
            max_tokens=report_context_max_tokens,
            label="report_content",
        )

        internal_system_messages = self._build_internal_system_messages(
            answer_instruction=answer_instruction,
            current_datetime_context=_build_current_datetime_context(),
            executor_results_context=executor_results_context,
            memory_summary=memory_summary if need_session_memory else None,
            knowledge_context=knowledge_context,
            route_context=route_context,
            mcp_context=mcp_context,
            traffic_context=traffic_context,
            service_context=service_context,
            report_context=report_context,
        )
        input_system_messages, input_non_system_messages = self._split_system_messages(
            input_messages
        )

        if not need_session_memory:
            context_messages = [
                *internal_system_messages,
                *input_system_messages,
                *input_non_system_messages,
            ]
            context_messages = limit_messages_to_recent_turns(
                context_messages,
                max_turns=max_turns,
            )
            context_messages = self._enforce_prompt_token_budget(
                context_messages,
                model_name=model_name,
            )
            return PreparedContext(
                messages=context_messages,
                used_session_memory=False,
                estimated_prompt_tokens=estimate_messages_tokens(
                    context_messages,
                    model_name=model_name,
                ),
                memory_summary=None,
                knowledge_context=knowledge_context,
                route_context=route_context,
                mcp_context=mcp_context,
                traffic_context=traffic_context,
                service_context=service_context,
                report_context=report_context,
                answer_instruction=answer_instruction,
                executor_results_context=executor_results_context,
            )

        recent_system_messages, recent_non_system_messages = self._split_system_messages(
            recent_messages
        )

        # Keep history system messages before non-system ones and remove duplicate suffixes.
        deduplicated_recent_messages = self._drop_overlapped_recent_suffix(
            recent_messages=recent_non_system_messages,
            input_messages=input_non_system_messages,
        )
        context_messages = [
            *internal_system_messages,
            *input_system_messages,
            *recent_system_messages,
            *deduplicated_recent_messages,
            *input_non_system_messages,
        ]
        context_messages = limit_messages_to_recent_turns(
            context_messages,
            max_turns=max_turns,
        )
        context_messages = self._enforce_prompt_token_budget(
            context_messages,
            model_name=model_name,
        )
        return PreparedContext(
            messages=context_messages,
            used_session_memory=bool(
                memory_summary or recent_system_messages or deduplicated_recent_messages
            ),
            estimated_prompt_tokens=estimate_messages_tokens(
                context_messages,
                model_name=model_name,
            ),
            memory_summary=memory_summary,
            knowledge_context=knowledge_context,
            route_context=route_context,
            mcp_context=mcp_context,
            traffic_context=traffic_context,
            service_context=service_context,
            report_context=report_context,
            answer_instruction=answer_instruction,
            executor_results_context=executor_results_context,
        )

    def _drop_overlapped_recent_suffix(
        self,
        *,
        recent_messages: Sequence[LlmInputMessage],
        input_messages: Sequence[LlmInputMessage],
    ) -> list[LlmInputMessage]:
        """Remove duplicated suffix messages from session history."""

        overlap_size = min(len(recent_messages), len(input_messages))
        while overlap_size > 0:
            if list(recent_messages[-overlap_size:]) == list(input_messages[-overlap_size:]):
                return list(recent_messages[:-overlap_size])
            overlap_size -= 1
        return list(recent_messages)
