"""知识服务模块。

负责编排本地数据集映射、RAGFlow 客户端与知识检索结果格式化。
当前阶段只实现最小可用知识库接入，不负责复杂权限控制和知识库写入流程。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppException
from app.knowledge.ragflow.chat import RagflowChatClient
from app.knowledge.ragflow.datasets import RagflowDatasetClient
from app.knowledge.ragflow.documents import RagflowDocumentClient
from app.knowledge.ragflow.retrieval import RagflowRetrievalClient
from app.persistence.ragflow_repo import RagflowRepository
from app.schemas.knowledge import (
    KnowledgeChatRequest,
    KnowledgeDatasetItem,
    KnowledgeDatasetListResponse,
    KnowledgeDocumentItem,
    KnowledgeDocumentListResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
)
from app.schemas.openai_compat import OpenAIChatCompletionResponse


class KnowledgeService:
    """知识库服务。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        dataset_client: RagflowDatasetClient | None = None,
        document_client: RagflowDocumentClient | None = None,
        retrieval_client: RagflowRetrievalClient | None = None,
        chat_client: RagflowChatClient | None = None,
    ) -> None:
        self._db_session = db_session
        self._ragflow_repository = RagflowRepository(db_session)
        self._dataset_client = dataset_client or RagflowDatasetClient()
        self._document_client = document_client or RagflowDocumentClient()
        self._retrieval_client = retrieval_client or RagflowRetrievalClient()
        self._chat_client = chat_client or RagflowChatClient()

    async def sync_datasets(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> KnowledgeDatasetListResponse:
        """同步远端 RAGFlow 数据集到本地映射表。"""

        remote_datasets = await self._dataset_client.list_datasets(page=page, page_size=page_size)
        synced_count = 0
        for remote_dataset in remote_datasets:
            dataset_id = str(
                remote_dataset.get("id")
                or remote_dataset.get("dataset_id")
                or remote_dataset.get("kb_id")
                or ""
            )
            if not dataset_id:
                continue

            dataset_name = str(
                remote_dataset.get("name")
                or remote_dataset.get("dataset_name")
                or remote_dataset.get("kb_name")
                or dataset_id
            )
            await self._ragflow_repository.upsert_dataset(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                is_enabled=self._resolve_dataset_enabled(remote_dataset),
                dataset_metadata=remote_dataset,
            )
            synced_count += 1

        await self._db_session.commit()
        dataset_response = await self.list_datasets()
        dataset_response.synced_count = synced_count
        return dataset_response

    async def list_datasets(
        self,
        *,
        is_enabled: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> KnowledgeDatasetListResponse:
        """查询本地维护的数据集映射列表。"""

        dataset_entities = await self._ragflow_repository.list_datasets(
            is_enabled=is_enabled,
            limit=limit,
            offset=offset,
        )
        return KnowledgeDatasetListResponse(
            items=[
                KnowledgeDatasetItem(
                    dataset_id=dataset_entity.dataset_id,
                    dataset_name=dataset_entity.dataset_name,
                    is_enabled=dataset_entity.is_enabled,
                    metadata=dataset_entity.dataset_metadata,
                )
                for dataset_entity in dataset_entities
            ]
        )

    async def list_documents(
        self,
        *,
        dataset_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> KnowledgeDocumentListResponse:
        """查询远端数据集下的文档列表。"""

        document_payloads = await self._document_client.list_documents(
            dataset_id=dataset_id,
            page=page,
            page_size=page_size,
        )
        return KnowledgeDocumentListResponse(
            items=[
                KnowledgeDocumentItem(
                    document_id=str(
                        document_payload.get("id")
                        or document_payload.get("document_id")
                        or document_payload.get("doc_id")
                        or ""
                    ),
                    document_name=str(
                        document_payload.get("name")
                        or document_payload.get("filename")
                        or document_payload.get("document_name")
                        or document_payload.get("id")
                        or ""
                    ),
                    status=str(document_payload.get("status") or "unknown"),
                    metadata=document_payload,
                )
                for document_payload in document_payloads
            ]
        )

    async def retrieve_knowledge(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResponse:
        """执行知识检索并返回标准化结果。"""

        dataset_ids = await self._resolve_dataset_ids(
            preferred_dataset_ids=request.dataset_ids,
            require_available=True,
        )
        return KnowledgeSearchResponse(
            results=await self._retrieve_results(
                query=request.query,
                dataset_ids=dataset_ids,
                top_k=request.top_k,
            )
        )

    async def retrieve_for_agent(
        self,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        """为 Agent 路由提供知识检索结果。

        这里默认尝试使用本地已启用的数据集；如果尚未配置任何数据集，则直接返回空结果，
        避免知识库链路把整条聊天请求硬中断。
        """

        dataset_ids = await self._resolve_dataset_ids(
            preferred_dataset_ids=[],
            require_available=False,
        )
        if not dataset_ids:
            return []
        return await self._retrieve_results(query=query, dataset_ids=dataset_ids, top_k=top_k)

    async def create_chat_completion(
        self,
        request: KnowledgeChatRequest,
    ) -> OpenAIChatCompletionResponse:
        """透传调用 RAGFlow chat assistant completion 接口。"""

        if request.stream:
            raise AppException(
                "当前阶段知识库聊天接口暂不支持 stream=true。",
                error_code="invalid_request",
            )

        latest_user_message = ""
        for message in reversed(request.messages):
            if message.role == "user" and message.content:
                latest_user_message = str(message.content)
                break
        if not latest_user_message:
            raise AppException(
                "知识库聊天请求至少需要一条非空 user 消息。",
                error_code="invalid_request",
            )

        raw_response = await self._chat_client.create_chat_completion(
            chat_id=request.chat_id,
            request_payload={
                "question": latest_user_message,
                "stream": False,
            },
        )
        return OpenAIChatCompletionResponse.model_validate(raw_response)

    async def _resolve_dataset_ids(
        self,
        *,
        preferred_dataset_ids: list[str],
        require_available: bool,
    ) -> list[str]:
        """解析本次请求应使用的数据集列表。"""

        if preferred_dataset_ids:
            return preferred_dataset_ids

        enabled_datasets = await self._ragflow_repository.list_datasets(is_enabled=True, limit=100)
        dataset_ids = [dataset.dataset_id for dataset in enabled_datasets]
        if dataset_ids or not require_available:
            return dataset_ids

        raise AppException(
            "当前未配置任何可用知识库数据集，请先同步并启用数据集。",
            error_code="knowledge_dataset_not_configured",
        )

    async def _retrieve_results(
        self,
        *,
        query: str,
        dataset_ids: list[str],
        top_k: int,
    ) -> list[KnowledgeSearchResult]:
        """执行一次检索并把原始结果标准化。"""

        retrieval_payloads = await self._retrieval_client.retrieve_chunks(
            query=query,
            dataset_ids=dataset_ids,
            top_k=top_k,
        )
        normalized_results: list[KnowledgeSearchResult] = []
        for retrieval_payload in retrieval_payloads:
            content = str(
                retrieval_payload.get("content")
                or retrieval_payload.get("text")
                or retrieval_payload.get("chunk")
                or retrieval_payload.get("chunk_text")
                or ""
            ).strip()
            if not content:
                continue

            normalized_results.append(
                KnowledgeSearchResult(
                    document_id=str(
                        retrieval_payload.get("document_id")
                        or retrieval_payload.get("doc_id")
                        or retrieval_payload.get("id")
                        or ""
                    ),
                    chunk_id=str(
                        retrieval_payload.get("chunk_id")
                        or retrieval_payload.get("id")
                        or retrieval_payload.get("doc_id")
                        or ""
                    ),
                    score=float(
                        retrieval_payload.get("score")
                        or retrieval_payload.get("similarity")
                        or retrieval_payload.get("similarity_score")
                        or 0.0
                    ),
                    content=content,
                    source=(
                        str(
                            retrieval_payload.get("source")
                            or retrieval_payload.get("document_name")
                            or retrieval_payload.get("filename")
                        )
                        if (
                            retrieval_payload.get("source")
                            or retrieval_payload.get("document_name")
                            or retrieval_payload.get("filename")
                        )
                        else None
                    ),
                )
            )
        return normalized_results

    @staticmethod
    def _resolve_dataset_enabled(remote_dataset: dict[str, object]) -> bool:
        """根据远端字段推导本地默认启用状态。"""

        status_value = remote_dataset.get("status")
        if status_value in {False, "0", "disabled", "inactive"}:
            return False
        enabled_value = remote_dataset.get("enabled")
        if isinstance(enabled_value, bool):
            return enabled_value
        return True
