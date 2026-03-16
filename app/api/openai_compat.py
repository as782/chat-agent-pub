"""OpenAI 兼容接口模块。
负责暴露 OpenAI Chat Completions 兼容路径，便于复用通用客户端和未来模型切换。
当前阶段不负责 Responses API 兼容和多会话持久化。
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.api.streaming import prime_stream_iterator
from app.schemas.openai_compat import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from app.services.openai_compat_service import OpenAICompatService

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@router.post(
    "/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_openai_chat_completion(
    request: OpenAIChatCompletionRequest,
) -> OpenAIChatCompletionResponse | StreamingResponse:
    """处理 OpenAI Chat Completions 兼容请求。"""

    compat_service = OpenAICompatService()
    if request.stream:
        stream_iterator = await compat_service.stream_chat_completion(request)
        primed_stream_iterator = await prime_stream_iterator(stream_iterator)
        return StreamingResponse(
            primed_stream_iterator,
            media_type="text/event-stream",
        )
    return await compat_service.create_chat_completion(request)
