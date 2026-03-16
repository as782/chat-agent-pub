"""知识库接口集成测试。"""

from fastapi.testclient import TestClient


def test_knowledge_api_syncs_datasets(app_client: TestClient, monkeypatch) -> None:
    """验证知识库接口可以同步远端数据集并返回本地映射。"""

    async def fake_list_datasets(
        self: object,
        *,
        page: int = 1,
        page_size: int = 50,
        keyword: str | None = None,
    ) -> list[dict[str, object]]:
        """返回稳定的远端数据集列表。"""

        del self, page, page_size, keyword
        return [
            {
                "id": "dataset-001",
                "name": "杭州旅游知识库",
                "status": "1",
            }
        ]

    monkeypatch.setattr(
        "app.knowledge.ragflow.datasets.RagflowDatasetClient.list_datasets",
        fake_list_datasets,
    )

    response = app_client.post("/api/v1/knowledge/datasets/sync")
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["synced_count"] == 1
    assert response_payload["items"][0]["dataset_id"] == "dataset-001"
    assert response_payload["items"][0]["dataset_name"] == "杭州旅游知识库"


def test_knowledge_api_retrieves_chunks(app_client: TestClient, monkeypatch) -> None:
    """验证知识库接口可以返回标准化检索结果。"""

    async def fake_retrieve_chunks(
        self: object,
        *,
        query: str,
        dataset_ids: list[str],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """返回稳定的检索结果。"""

        del self, query, dataset_ids, top_k
        return [
            {
                "document_id": "doc-001",
                "chunk_id": "chunk-001",
                "score": 0.98,
                "content": "西湖位于浙江省杭州市。",
                "document_name": "杭州百科",
            }
        ]

    async def fake_list_datasets(
        self: object,
        *,
        page: int = 1,
        page_size: int = 50,
        keyword: str | None = None,
    ) -> list[dict[str, object]]:
        """返回稳定的远端数据集列表。"""

        del self, page, page_size, keyword
        return [{"id": "dataset-001", "name": "杭州旅游知识库"}]

    monkeypatch.setattr(
        "app.knowledge.ragflow.datasets.RagflowDatasetClient.list_datasets",
        fake_list_datasets,
    )
    monkeypatch.setattr(
        "app.knowledge.ragflow.retrieval.RagflowRetrievalClient.retrieve_chunks",
        fake_retrieve_chunks,
    )

    app_client.post("/api/v1/knowledge/datasets/sync")
    response = app_client.post(
        "/api/v1/knowledge/retrieval",
        json={"query": "西湖在哪里", "top_k": 3},
    )
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["results"][0]["document_id"] == "doc-001"
    assert response_payload["results"][0]["content"] == "西湖位于浙江省杭州市。"


def test_knowledge_api_returns_400_when_dataset_not_configured(app_client: TestClient) -> None:
    """验证未配置任何知识库数据集时会返回明确错误。"""

    response = app_client.post(
        "/api/v1/knowledge/retrieval",
        json={"query": "西湖在哪里", "top_k": 3},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "knowledge_dataset_not_configured"
