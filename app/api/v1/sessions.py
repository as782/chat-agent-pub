"""会话接口模块。

负责创建、查询和列出会话，并将业务逻辑委托给会话服务。
当前阶段不负责会话删除和复杂筛选能力。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_db_session
from app.schemas.session import SessionCreateRequest, SessionListResponse, SessionResponse
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreateRequest,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> SessionResponse:
    """创建会话。"""

    session_service = SessionService(db_session)
    return await session_service.create_session(request)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    user_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SessionListResponse:
    """查询会话列表。"""

    session_service = SessionService(db_session)
    return await session_service.list_sessions(user_id=user_id, limit=limit, offset=offset)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> SessionResponse:
    """查询单个会话。"""

    session_service = SessionService(db_session)
    return await session_service.get_session(session_id)
