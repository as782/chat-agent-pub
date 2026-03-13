"""消息接口模块。

负责查询会话消息历史并返回结构化结果。
当前阶段不负责消息新增、编辑和删除接口。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_db_session
from app.schemas.message import MessageListResponse
from app.services.message_service import MessageService

router = APIRouter(prefix="/messages", tags=["messages"])


@router.get("/{session_id}", response_model=MessageListResponse)
async def list_session_messages(
    session_id: str,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> MessageListResponse:
    """查询指定会话的消息历史。"""

    message_service = MessageService(db_session)
    return await message_service.list_messages(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
