"""OpenAI 兼容接口模块。

负责暴露 OpenAI Chat Completions 兼容路径，便于复用通用客户端和未来模型切换。
当前阶段不负责流式输出、函数调用和 Responses API 兼容。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_db_session
from app.schemas.openai_compat import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from app.services.openai_compat_service import OpenAICompatService

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@router.post(
    "/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_openai_chat_completion(
    request: OpenAIChatCompletionRequest,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> OpenAIChatCompletionResponse:
    """处理 OpenAI Chat Completions 兼容请求。"""

    compat_service = OpenAICompatService(db_session)
    return await compat_service.create_chat_completion(request)
