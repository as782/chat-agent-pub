"""RAGFlow 数据集客户端模块。

负责封装数据集列表相关接口，供知识服务同步和查询本地映射时复用。
当前阶段不负责数据集创建、删除和权限管理。
"""

from __future__ import annotations

from typing import Any

from app.knowledge.ragflow.client import RagflowClient


class RagflowDatasetClient:
    """RAGFlow 数据集客户端。"""

    def __init__(self, ragflow_client: RagflowClient | None = None) -> None:
        self._ragflow_client = ragflow_client or RagflowClient()

    async def list_datasets(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询远端数据集列表。"""

        response_data = await self._ragflow_client.request(
            "GET",
            "/api/v1/datasets",
            params={
                "page": page,
                "page_size": page_size,
                "keyword": keyword,
            },
        )
        return self._normalize_list_payload(response_data)

    @staticmethod
    def _normalize_list_payload(response_data: Any) -> list[dict[str, Any]]:
        """兼容不同版本的列表字段命名。"""

        if isinstance(response_data, list):
            return [dataset for dataset in response_data if isinstance(dataset, dict)]
        if not isinstance(response_data, dict):
            return []

        for field_name in ("docs", "items", "datasets", "list"):
            field_value = response_data.get(field_name)
            if isinstance(field_value, list):
                return [dataset for dataset in field_value if isinstance(dataset, dict)]
        return []
