"""上下文构建模块。
负责把显式请求消息、会话历史和记忆摘要整合为模型可消费的上下文。
当前阶段不负责复杂重排、向量召回和提示词压缩优化。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache
from json import dumps
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from app.agent.prompts import (
    CURRENT_DATETIME_CONTEXT_PROMPT_PREFIX,
    MEMORY_SUMMARY_PROMPT_PREFIX,
)
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


def _build_current_datetime_context(timezone_name: str = "Asia/Shanghai") -> str:
    """构建当前时间上下文，便于回答涉及今天、明天、节假日窗口的问题。"""

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


def _convert_input_message_to_langchain_message(message: LlmInputMessage) -> BaseMessage:
    """灏?LlmInputMessage 杞崲涓?LangChain message锛屼究浜庤绠楀厜鍙ｆ暟銆?"""

    normalized_content = message.content.strip()
    if message.role == "user":
        return HumanMessage(content=normalized_content, name=message.name)
    if message.role == "assistant":
        return AIMessage(
            content=normalized_content,
            name=message.name,
            tool_calls=[
                {
                    "name": tool_call.tool_name,
                    "args": tool_call.arguments,
                    "id": tool_call.tool_call_id,
                    "type": "tool_call",
                }
                for tool_call in message.tool_calls
            ],
        )
    if message.role == "system":
        return SystemMessage(content=normalized_content, name=message.name)
    if message.role == "tool":
        return ToolMessage(
            content=normalized_content,
            tool_call_id=message.tool_call_id or "",
        )
    return ChatMessage(role=message.role, content=normalized_content, name=message.name)


@lru_cache(maxsize=32)
def _get_token_counter_model(model_name: str) -> ChatOpenAI:
    """鍒涘缓涓€涓粎鐢ㄤ簬璁＄畻 token 鐨勫鎴风瀵硅薄銆?"""

    return ChatOpenAI(model=model_name, api_key="token-counter")


def _get_token_encoding(model_name: str) -> tiktoken.Encoding:
    """鑾峰彇鍙敤鐨勬柟璦€旇█缂栫爜銆?"""

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        for encoding_name in ("o200k_base", "cl100k_base"):
            try:
                return tiktoken.get_encoding(encoding_name)
            except Exception:
                continue
        return tiktoken.get_encoding("cl100k_base")


def estimate_messages_tokens(
    messages: Sequence[LlmInputMessage],
    *,
    model_name: str | None = None,
) -> int:
    """浼拌杈撳叆 messages 鐨勬秷鑰禝 token 鏁般€?"""

    normalized_model_name = (model_name or "gpt-4o-mini").strip() or "gpt-4o-mini"
    langchain_messages = [
        _convert_input_message_to_langchain_message(message) for message in messages
    ]

    try:
        return int(
            _get_token_counter_model(normalized_model_name).get_num_tokens_from_messages(
                langchain_messages
            )
        )
    except Exception:
        encoding = _get_token_encoding(normalized_model_name)
        total_tokens = 0
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
        return total_tokens + 3


class ContextBuilder:
    """对话上下文构建器。"""

    @staticmethod
    def _split_system_messages(
        messages: Sequence[LlmInputMessage],
    ) -> tuple[list[LlmInputMessage], list[LlmInputMessage]]:
        """拆分 system 与非 system 消息，确保最终上下文满足提供方顺序约束。"""

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
        """构建由系统内部注入的 system 消息。"""

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
        """构建当前轮次的模型输入上下文。

        设计约束：
        - 不带 session_id 的请求不注入系统记忆，只使用本次显式 messages。
        - 带 session_id 的请求同时参考系统历史和本次显式 messages。
        """

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

        # 历史非 system 消息放前，本次显式非 system 输入放后，同时去掉完全重叠的尾部。
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
        """删除系统历史尾部与本次输入重复的后缀，避免上下文中出现连续重复消息。"""

        overlap_size = min(len(recent_messages), len(input_messages))
        while overlap_size > 0:
            if list(recent_messages[-overlap_size:]) == list(input_messages[-overlap_size:]):
                return list(recent_messages[:-overlap_size])
            overlap_size -= 1
        return list(recent_messages)
