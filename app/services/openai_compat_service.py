"""OpenAI 兼容适配服务模块。

负责将 OpenAI Chat Completions 兼容请求转换为当前系统的对话调用。
当前阶段不负责流式输出、工具调用和多轮会话持久化对齐。
"""

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppException
from app.schemas.openai_compat import (
    OpenAIChatCompletionAssistantMessage,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionUsage,
    OpenAIChatMessage,
)
from app.services.chat_service import ChatService


class OpenAICompatService:
    """OpenAI 兼容接口适配服务。"""

    def __init__(self, db_session: AsyncSession, chat_service: ChatService | None = None) -> None:
        self._chat_service = chat_service or ChatService(db_session)

    async def create_chat_completion(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> OpenAIChatCompletionResponse:
        """处理 OpenAI 兼容聊天请求。"""

        if request.stream:
            raise AppException(
                "当前适配层暂不支持 stream=true。",
                error_code="unsupported_feature",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        prompt_messages = self._build_prompt_messages(request.messages)
        latest_user_message = self._extract_latest_user_message(request.messages)
        turn_result = await self._chat_service.send_prompt_messages(
            prompt_messages=prompt_messages,
            latest_user_message=latest_user_message,
            model_name=request.model,
            user_id=request.user,
        )

        return OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(datetime.now(UTC).timestamp()),
            model=turn_result.model_name,
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=OpenAIChatCompletionAssistantMessage(content=turn_result.answer),
                    finish_reason=turn_result.finish_reason,
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=turn_result.prompt_tokens,
                completion_tokens=turn_result.completion_tokens,
                total_tokens=turn_result.total_tokens,
            ),
        )

    def _build_prompt_messages(self, messages: list[OpenAIChatMessage]) -> list[tuple[str, str]]:
        """将 OpenAI 兼容消息转换为内部提示词消息。"""

        prompt_messages: list[tuple[str, str]] = []
        for message in messages:
            normalized_content = self._normalize_message_content(message)
            prompt_messages.append((message.role, normalized_content))

        return prompt_messages

    def _extract_latest_user_message(self, messages: list[OpenAIChatMessage]) -> str:
        """提取最后一条用户消息，供当前系统做会话摘要和落库。"""

        for message in reversed(messages):
            if message.role == "user":
                normalized_content = self._normalize_message_content(message)
                if normalized_content:
                    return normalized_content

        raise AppException(
            "OpenAI 兼容请求至少需要包含一条非空 user 消息。",
            error_code="invalid_request",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    @staticmethod
    def _normalize_message_content(message: OpenAIChatMessage) -> str:
        """将消息内容统一转换为纯文本。"""

        content = message.content
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = [
                content_part.text.strip() for content_part in content if content_part.text.strip()
            ]
            return "\n".join(text_parts).strip()

        return ""
