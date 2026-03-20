"""KnowledgeService 单元测试。"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import UpstreamServiceException
from app.knowledge.service import KnowledgeService
from app.persistence.ragflow_repo import RagflowRepository
from app.schemas.knowledge import KnowledgeSearchRequest


class _MixedModelFallbackRetrievalClient:
    """模拟多数据集联合检索报 embedding 冲突、单数据集检索可用。"""

    async def retrieve_chunks(
        self,
        *,
        query: str,
        dataset_ids: list[str],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        del query, top_k
        if len(dataset_ids) > 1:
            raise UpstreamServiceException(
                "Datasets use different embedding models.",
                error_code="ragflow_business_error",
                details={
                    "path": "/api/v1/retrieval",
                    "response": {
                        "code": 102,
                        "message": "Datasets use different embedding models.\"",
                    },
                },
            )
        if dataset_ids == ["dataset-a"]:
            return [
                {"document_id": "doc-a", "chunk_id": "chunk-a", "score": 0.61, "content": "A"},
                {"document_id": "doc-a2", "chunk_id": "chunk-a2", "score": 0.44, "content": "A2"},
            ]
        if dataset_ids == ["dataset-b"]:
            return [
                {"document_id": "doc-b", "chunk_id": "chunk-b", "score": 0.95, "content": "B"},
            ]
        return []


class _AlwaysFailRetrievalClient:
    """模拟非 embedding 冲突场景，确保异常按原样抛出。"""

    async def retrieve_chunks(
        self,
        *,
        query: str,
        dataset_ids: list[str],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        del query, dataset_ids, top_k
        raise UpstreamServiceException(
            "RAGFlow internal error",
            error_code="ragflow_business_error",
            details={"path": "/api/v1/retrieval", "response": {"code": 500, "message": "internal"}},
        )


class _CaptureDatasetRetrievalClient:
    """捕获检索入参，验证默认数据集 ID 生效。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def retrieve_chunks(
        self,
        *,
        query: str,
        dataset_ids: list[str],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        del query, top_k
        self.calls.append(list(dataset_ids))
        return [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "score": 0.9,
                "content": "single dataset result",
            }
        ]


@pytest.mark.asyncio
async def test_knowledge_service_falls_back_to_per_dataset_retrieval_when_embedding_mixed(
    db_session: AsyncSession,
) -> None:
    """验证 embedding 模型冲突时会自动降级逐数据集检索并合并结果。"""

    repository = RagflowRepository(db_session)
    await repository.upsert_dataset(dataset_id="dataset-a", dataset_name="A", is_enabled=True)
    await repository.upsert_dataset(dataset_id="dataset-b", dataset_name="B", is_enabled=True)
    await db_session.commit()

    service = KnowledgeService(
        db_session,
        retrieval_client=_MixedModelFallbackRetrievalClient(),
        settings=Settings(DEFAULT_KNOWLEDGE_DATASET_ID=""),
    )
    response = await service.retrieve_knowledge(
        KnowledgeSearchRequest(query="test query", top_k=2),
    )

    assert [item.document_id for item in response.results] == ["doc-b", "doc-a"]
    assert [item.score for item in response.results] == [0.95, 0.61]


@pytest.mark.asyncio
async def test_knowledge_service_keeps_raising_non_mixed_embedding_errors(
    db_session: AsyncSession,
) -> None:
    """验证非 embedding 冲突错误不会被错误吞掉。"""

    repository = RagflowRepository(db_session)
    await repository.upsert_dataset(dataset_id="dataset-a", dataset_name="A", is_enabled=True)
    await db_session.commit()

    service = KnowledgeService(
        db_session,
        retrieval_client=_AlwaysFailRetrievalClient(),
        settings=Settings(DEFAULT_KNOWLEDGE_DATASET_ID=""),
    )

    with pytest.raises(UpstreamServiceException, match="RAGFlow internal error"):
        await service.retrieve_knowledge(
            KnowledgeSearchRequest(query="test query", top_k=2),
        )


@pytest.mark.asyncio
async def test_knowledge_service_uses_default_dataset_id_from_settings(
    db_session: AsyncSession,
) -> None:
    """验证配置了 DEFAULT_KNOWLEDGE_DATASET_ID 时始终只检索该单一数据集。"""

    capture_client = _CaptureDatasetRetrievalClient()
    settings = Settings(
        DEFAULT_KNOWLEDGE_DATASET_ID="dataset-fixed",
    )
    service = KnowledgeService(
        db_session,
        retrieval_client=capture_client,
        settings=settings,
    )

    await service.retrieve_knowledge(
        KnowledgeSearchRequest(
            query="query",
            # 即使显式传入其他 dataset_ids，也应以配置的单一数据集为准
            dataset_ids=["dataset-other-a", "dataset-other-b"],
            top_k=3,
        ),
    )

    assert capture_client.calls == [["dataset-fixed"]]
