"""上下文构建模块。
负责把显式请求消息、会话历史和记忆摘要整合为模型可消费的上下文。
当前阶段不负责复杂重排、向量召回和提示词压缩优化。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.state import PreparedContext
from app.clients.llm_client import LlmInputMessage, LlmToolCall
from app.persistence.models import MessageEntity

MEMORY_SUMMARY_PROMPT_PREFIX = "以下是当前会话的历史摘要，仅在不与用户本次显式输入冲突时参考：\n"


def message_entity_to_input_message(message_entity: MessageEntity) -> LlmInputMessage:
    """把消息实体转换为统一输入消息。"""

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
    """序列化统一消息，供记忆仓储和 Redis checkpoint 复用。"""

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
    """反序列化统一消息，恢复 checkpoint 中保存的上下文窗口。"""

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


class ContextBuilder:
    """对话上下文构建器。"""

    def build_context(
        self,
        *,
        input_messages: Sequence[LlmInputMessage],
        recent_messages: Sequence[LlmInputMessage],
        memory_summary: str | None,
        need_session_memory: bool,
        knowledge_context: str | None = None,
    ) -> PreparedContext:
        """构建当前轮次的模型输入上下文。

        设计约束：
        - 不带 session_id 的请求不注入系统记忆，只使用本次显式 messages。
        - 带 session_id 的请求同时参考系统历史和本次显式 messages。
        """

        if not need_session_memory:
            context_messages = []
            if knowledge_context:
                context_messages.append(
                    LlmInputMessage(
                        role="system",
                        content=knowledge_context,
                    )
                )
            context_messages.extend(input_messages)
            return PreparedContext(
                messages=context_messages,
                used_session_memory=False,
                memory_summary=None,
                knowledge_context=knowledge_context,
            )

        context_messages: list[LlmInputMessage] = []
        if memory_summary:
            context_messages.append(
                LlmInputMessage(
                    role="system",
                    content=f"{MEMORY_SUMMARY_PROMPT_PREFIX}{memory_summary}",
                )
            )
        if knowledge_context:
            context_messages.append(
                LlmInputMessage(
                    role="system",
                    content=knowledge_context,
                )
            )

        # 这里按“系统历史在前、本次显式输入在后”合并上下文，
        # 同时移除历史尾部与本次输入尾部完全重叠的部分，避免重复注入最后一轮用户消息。
        deduplicated_recent_messages = self._drop_overlapped_recent_suffix(
            recent_messages=recent_messages,
            input_messages=input_messages,
        )
        context_messages.extend(deduplicated_recent_messages)
        context_messages.extend(input_messages)
        return PreparedContext(
            messages=context_messages,
            used_session_memory=bool(memory_summary or deduplicated_recent_messages),
            memory_summary=memory_summary,
            knowledge_context=knowledge_context,
        )

    def _drop_overlapped_recent_suffix(
        self,
        *,
        recent_messages: Sequence[LlmInputMessage],
        input_messages: Sequence[LlmInputMessage],
    ) -> list[LlmInputMessage]:
        """删除系统历史尾部与本次输入重复的后缀，避免上下文中出现连续重复消息。"""

        overlap_size = min(len(recent_messages), len(input_messages))
        while overlap_size > 0:
            if list(recent_messages[-overlap_size:]) == list(input_messages[-overlap_size:]):
                return list(recent_messages[:-overlap_size])
            overlap_size -= 1
        return list(recent_messages)
