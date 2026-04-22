"""Context building utilities.

This module assembles the final message list sent to the model and provides a
safe token estimation helper for logging and monitoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from json import dumps
import math
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import tiktoken

from app.agent.prompts import (
    CURRENT_DATETIME_CONTEXT_PROMPT_PREFIX,
    MEMORY_SUMMARY_PROMPT_PREFIX,
)
from app.agent.history_utils import limit_messages_to_recent_turns
from app.agent.state import PreparedContext
from app.clients.llm_client import LlmInputMessage, LlmToolCall
from app.persistence.models import MessageEntity


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

    def build_context(
        self,
        *,
        input_messages: Sequence[LlmInputMessage],
        recent_messages: Sequence[LlmInputMessage],
        memory_summary: str | None,
        need_session_memory: bool,
        model_name: str | None = None,
        answer_instruction: str | None = None,
        executor_results_context: str | None = None,
        knowledge_context: str | None = None,
        route_context: str | None = None,
        mcp_context: str | None = None,
        traffic_context: str | None = None,
        service_context: str | None = None,
        report_context: str | None = None,
    ) -> PreparedContext:
        """Build the final model input for the current turn."""

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
        context_messages = limit_messages_to_recent_turns(context_messages)
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
