"""记忆摘要模块。
负责把较长的会话历史压缩成可读摘要，并保留最近上下文窗口。
当前阶段使用规则摘要，避免额外模型调用成本；后续可替换为 LLM 摘要器。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.agent.context_builder import message_entity_to_input_message
from app.agent.history_utils import MAX_CONTEXT_MESSAGES
from app.clients.llm_client import LlmInputMessage
from app.persistence.models import MessageEntity

DEFAULT_CONTEXT_WINDOW_SIZE = MAX_CONTEXT_MESSAGES
DEFAULT_SUMMARY_TRIGGER_COUNT = MAX_CONTEXT_MESSAGES
DEFAULT_SUMMARY_LINE_LIMIT = 8


@dataclass(slots=True)
class MemorySummaryResult:
    """记忆摘要结果。"""

    summary: str | None
    context_window_messages: list[LlmInputMessage]
    message_count: int


class MemorySummarizer:
    """最小可用记忆摘要器。"""

    def __init__(
        self,
        *,
        context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE,
        summary_trigger_count: int = DEFAULT_SUMMARY_TRIGGER_COUNT,
        summary_line_limit: int = DEFAULT_SUMMARY_LINE_LIMIT,
    ) -> None:
        self._context_window_size = context_window_size
        self._summary_trigger_count = summary_trigger_count
        self._summary_line_limit = summary_line_limit

    def summarize_messages(self, messages: Sequence[MessageEntity]) -> MemorySummaryResult:
        """从会话消息生成摘要和最近上下文窗口。"""

        normalized_messages = [message_entity_to_input_message(message) for message in messages]
        message_count = len(normalized_messages)
        context_window_messages = normalized_messages[-self._context_window_size :]

        if message_count <= self._summary_trigger_count:
            return MemorySummaryResult(
                summary=None,
                context_window_messages=context_window_messages,
                message_count=message_count,
            )

        older_messages = normalized_messages[: -self._context_window_size] or normalized_messages
        summary_lines: list[str] = []
        for message in older_messages[-self._summary_line_limit :]:
            summary_lines.append(
                f"{self._resolve_role_name(message.role)}：{self._build_message_preview(message)}"
            )

        summary = "以下是更早会话摘要，请仅在不与用户当前显式输入冲突时参考：\n" + "\n".join(
            summary_lines
        )
        return MemorySummaryResult(
            summary=summary,
            context_window_messages=context_window_messages,
            message_count=message_count,
        )

    @staticmethod
    def _resolve_role_name(role: str) -> str:
        """把内部角色名转换为更可读的中文标签。"""

        return {
            "user": "用户",
            "assistant": "助手",
            "system": "系统",
            "tool": "工具",
        }.get(role, role)

    @staticmethod
    def _build_message_preview(message: LlmInputMessage) -> str:
        """构建摘要中使用的消息预览文本。"""

        if message.tool_calls:
            tool_names = ",".join(
                tool_call.tool_name for tool_call in message.tool_calls if tool_call.tool_name
            )
            return f"请求调用工具：{tool_names}"

        normalized_content = message.content.strip().replace("\n", " ")
        if len(normalized_content) > 80:
            return f"{normalized_content[:80]}..."
        return normalized_content
