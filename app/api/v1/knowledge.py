"""知识库接口模块。

负责暴露数据集同步、文档查询、知识检索和 RAGFlow 聊天透传接口。
当前阶段不负责复杂知识库配置管理和权限控制。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.service import KnowledgeService
from app.persistence.database import get_db_session
from app.schemas.knowledge import (
    KnowledgeChatRequest,
    KnowledgeDatasetListResponse,
    KnowledgeDocumentListResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from app.schemas.openai_compat import OpenAIChatCompletionResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post(
    "/datasets/sync",
    response_model=KnowledgeDatasetListResponse,
    status_code=status.HTTP_200_OK,
)
async def sync_ragflow_datasets(
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> KnowledgeDatasetListResponse:
    """同步远端 RAGFlow 数据集到本地映射表。"""

    knowledge_service = KnowledgeService(db_session)
    return await knowledge_service.sync_datasets()


@router.get(
    "/datasets",
    response_model=KnowledgeDatasetListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_knowledge_datasets(
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    is_enabled: bool | None = Query(default=None, description="按启用状态过滤。"),
    limit: int = Query(default=50, ge=1, le=100, description="分页大小。"),
    offset: int = Query(default=0, ge=0, description="分页偏移。"),
) -> KnowledgeDatasetListResponse:
    """查询本地已维护的知识库数据集映射。"""

    knowledge_service = KnowledgeService(db_session)
    return await knowledge_service.list_datasets(
        is_enabled=is_enabled,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/datasets/{dataset_id}/documents",
    response_model=KnowledgeDocumentListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_knowledge_documents(
    dataset_id: str,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    page: int = Query(default=1, ge=1, description="页码。"),
    page_size: int = Query(default=20, ge=1, le=100, description="页大小。"),
) -> KnowledgeDocumentListResponse:
    """查询指定知识库数据集下的文档列表。"""

    knowledge_service = KnowledgeService(db_session)
    return await knowledge_service.list_documents(
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/retrieval",
    response_model=KnowledgeSearchResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve_knowledge(
    request: KnowledgeSearchRequest,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> KnowledgeSearchResponse:
    """执行一次知识检索。"""

    knowledge_service = KnowledgeService(db_session)
    return await knowledge_service.retrieve_knowledge(request)


@router.post(
    "/chat",
    response_model=OpenAIChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_knowledge_chat_completion(
    request: KnowledgeChatRequest,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> OpenAIChatCompletionResponse:
    """透传调用 RAGFlow chat assistant completion 接口。"""

    knowledge_service = KnowledgeService(db_session)
    return await knowledge_service.create_chat_completion(request)
