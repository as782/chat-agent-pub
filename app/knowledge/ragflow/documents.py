"""RAGFlow 文档客户端模块。

负责封装数据集文档列表接口，供知识库管理 API 查询远端文档概况。
当前阶段不负责文档上传、切分和解析任务控制。
"""

from __future__ import annotations

from typing import Any

from app.knowledge.ragflow.client import RagflowClient


class RagflowDocumentClient:
    """RAGFlow 文档客户端。"""

    def __init__(self, ragflow_client: RagflowClient | None = None) -> None:
        self._ragflow_client = ragflow_client or RagflowClient()

    async def list_documents(
        self,
        *,
        dataset_id: str,
        page: int = 1,
        page_size: int = 20,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询指定数据集下的文档列表。"""

        response_data = await self._ragflow_client.request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents",
            params={
                "page": page,
                "page_size": page_size,
                "keyword": keyword,
            },
        )
        return self._normalize_list_payload(response_data)

    @staticmethod
    def _normalize_list_payload(response_data: Any) -> list[dict[str, Any]]:
        """兼容不同版本的文档列表字段命名。"""

        if isinstance(response_data, list):
            return [document for document in response_data if isinstance(document, dict)]
        if not isinstance(response_data, dict):
            return []

        for field_name in ("docs", "items", "documents", "list"):
            field_value = response_data.get(field_name)
            if isinstance(field_value, list):
                return [document for document in field_value if isinstance(document, dict)]
        return []
