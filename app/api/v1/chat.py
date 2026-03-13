"""对话接口模块。

负责接收基础单轮对话请求并调用对话服务。
当前阶段不负责多轮状态图编排、工具调用和知识库路由。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_db_session
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def create_chat_completion(
    request: ChatRequest,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChatResponse:
    """处理基础单轮对话请求。"""

    chat_service = ChatService(db_session)
    return await chat_service.send_message(request)
