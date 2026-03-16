"""RAGFlow 检索客户端模块。

负责封装基于数据集的检索接口，把用户问题发送到 RAGFlow 并取回候选片段。
当前阶段不负责重排策略和多轮查询扩展。
"""

from __future__ import annotations

from typing import Any

from app.knowledge.ragflow.client import RagflowClient


class RagflowRetrievalClient:
    """RAGFlow 检索客户端。"""

    def __init__(self, ragflow_client: RagflowClient | None = None) -> None:
        self._ragflow_client = ragflow_client or RagflowClient()

    async def retrieve_chunks(
        self,
        *,
        query: str,
        dataset_ids: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """向 RAGFlow 发起知识检索请求。"""

        response_data = await self._ragflow_client.request(
            "POST",
            "/api/v1/retrieval",
            json_body={
                "question": query,
                "dataset_ids": dataset_ids,
                "top_k": top_k,
                "page_size": top_k,
            },
        )
        return self._normalize_list_payload(response_data)

    @staticmethod
    def _normalize_list_payload(response_data: Any) -> list[dict[str, Any]]:
        """兼容不同版本的检索结果字段命名。"""

        if isinstance(response_data, list):
            return [item for item in response_data if isinstance(item, dict)]
        if not isinstance(response_data, dict):
            return []

        for field_name in ("chunks", "items", "docs", "records", "list"):
            field_value = response_data.get(field_name)
            if isinstance(field_value, list):
                return [item for item in field_value if isinstance(item, dict)]
        return []
