"""OpenAI 兼容接口模块。
负责暴露 OpenAI Chat Completions 兼容路径，并复用内部聊天主链路。
当前阶段不负责 Responses API 兼容。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.streaming import prime_stream_iterator
from app.persistence.database import get_db_session
from app.schemas.openai_compat import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@router.post(
    "/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_openai_chat_completion(
    request: OpenAIChatCompletionRequest,
    response: Response,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
) -> OpenAIChatCompletionResponse | StreamingResponse:
    """处理 OpenAI Chat Completions 兼容请求。"""

    chat_service = ChatService(db_session)
    if request.stream:
        resolved_session_id, stream_iterator = await chat_service.stream_message(
            request,
            session_id=session_id,
        )
        primed_stream_iterator = await prime_stream_iterator(stream_iterator)
        return StreamingResponse(
            primed_stream_iterator,
            media_type="text/event-stream",
            headers={"X-Session-ID": resolved_session_id},
        )
    resolved_session_id, chat_response = await chat_service.send_message(
        request,
        session_id=session_id,
    )
    response.headers["X-Session-ID"] = resolved_session_id
    return chat_response
