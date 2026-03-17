"""上下文构建模块。
负责把显式请求消息、会话历史和记忆摘要整合为模型可消费的上下文。
当前阶段不负责复杂重排、向量召回和提示词压缩优化。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.prompts import MEMORY_SUMMARY_PROMPT_PREFIX
from app.agent.state import PreparedContext
from app.clients.llm_client import LlmInputMessage, LlmToolCall
from app.persistence.models import MessageEntity


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
    """序列化统一消息，供记忆存储和 Redis checkpoint 复用。"""

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
        answer_instruction: str | None = None,
        executor_results_context: str | None = None,
        knowledge_context: str | None = None,
        route_context: str | None = None,
        mcp_context: str | None = None,
        traffic_context: str | None = None,
        report_context: str | None = None,
    ) -> PreparedContext:
        """构建当前轮次的模型输入上下文。

        设计约束：
        - 不带 session_id 的请求不注入系统记忆，只使用本次显式 messages。
        - 带 session_id 的请求同时参考系统历史和本次显式 messages。
        """

        if not need_session_memory:
            context_messages: list[LlmInputMessage] = []
            if answer_instruction:
                context_messages.append(LlmInputMessage(role="system", content=answer_instruction))
            if executor_results_context:
                context_messages.append(
                    LlmInputMessage(role="system", content=executor_results_context)
                )
            if knowledge_context:
                context_messages.append(LlmInputMessage(role="system", content=knowledge_context))
            if route_context:
                context_messages.append(LlmInputMessage(role="system", content=route_context))
            if mcp_context:
                context_messages.append(LlmInputMessage(role="system", content=mcp_context))
            if traffic_context:
                context_messages.append(LlmInputMessage(role="system", content=traffic_context))
            if report_context:
                context_messages.append(LlmInputMessage(role="system", content=report_context))
            context_messages.extend(input_messages)
            return PreparedContext(
                messages=context_messages,
                used_session_memory=False,
                memory_summary=None,
                knowledge_context=knowledge_context,
                route_context=route_context,
                mcp_context=mcp_context,
                traffic_context=traffic_context,
                report_context=report_context,
                answer_instruction=answer_instruction,
                executor_results_context=executor_results_context,
            )

        context_messages: list[LlmInputMessage] = []
        if answer_instruction:
            context_messages.append(LlmInputMessage(role="system", content=answer_instruction))
        if executor_results_context:
            context_messages.append(
                LlmInputMessage(role="system", content=executor_results_context)
            )
        if memory_summary:
            context_messages.append(
                LlmInputMessage(
                    role="system",
                    content=f"{MEMORY_SUMMARY_PROMPT_PREFIX}{memory_summary}",
                )
            )
        if knowledge_context:
            context_messages.append(LlmInputMessage(role="system", content=knowledge_context))
        if route_context:
            context_messages.append(LlmInputMessage(role="system", content=route_context))
        if mcp_context:
            context_messages.append(LlmInputMessage(role="system", content=mcp_context))
        if traffic_context:
            context_messages.append(LlmInputMessage(role="system", content=traffic_context))
        if report_context:
            context_messages.append(LlmInputMessage(role="system", content=report_context))

        # 历史消息放前，本次显式输入放后，同时去掉完全重叠的尾部。
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
            route_context=route_context,
            mcp_context=mcp_context,
            traffic_context=traffic_context,
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
        """删除系统历史尾部与本次输入重复的后缀，避免上下文中出现连续重复消息。"""

        overlap_size = min(len(recent_messages), len(input_messages))
        while overlap_size > 0:
            if list(recent_messages[-overlap_size:]) == list(input_messages[-overlap_size:]):
                return list(recent_messages[:-overlap_size])
            overlap_size -= 1
        return list(recent_messages)
