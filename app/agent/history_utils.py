"""Conversation history helpers."""

from __future__ import annotations

from collections.abc import Sequence

from app.clients.llm_client import LlmInputMessage

MAX_CONTEXT_TURNS = 10
MAX_CONTEXT_MESSAGES = MAX_CONTEXT_TURNS * 2


def limit_messages_to_recent_turns(
    messages: Sequence[LlmInputMessage],
    *,
    max_turns: int = MAX_CONTEXT_TURNS,
) -> list[LlmInputMessage]:
    """Keep the newest `max_turns` user turns and any system messages."""

    if max_turns <= 0:
        return []

    user_turn_count = 0
    cutoff_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role != "user":
            continue
        user_turn_count += 1
        if user_turn_count == max_turns:
            cutoff_index = index
            break

    if cutoff_index is None:
        return list(messages)

    return [
        message
        for index, message in enumerate(messages)
        if index >= cutoff_index or message.role == "system"
    ]
