"""RAGFlow 客户端单元测试。"""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from app.knowledge.ragflow.chat import RagflowChatClient
from app.knowledge.ragflow.datasets import RagflowDatasetClient
from app.knowledge.ragflow.documents import RagflowDocumentClient
from app.knowledge.ragflow.retrieval import RagflowRetrievalClient


@pytest.mark.asyncio
async def test_ragflow_dataset_client_assembles_list_request(monkeypatch: MonkeyPatch) -> None:
    """验证数据集客户端会按约定组装列表查询参数。"""

    captured_call: dict[str, object] = {}

    async def fake_request(
        self: object,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
        expect_envelope: bool = True,
    ) -> dict[str, object]:
        """记录请求参数并返回模拟数据。"""

        del self
        captured_call.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
                "expect_envelope": expect_envelope,
            }
        )
        return {"items": []}

    monkeypatch.setattr("app.knowledge.ragflow.client.RagflowClient.request", fake_request)

    dataset_client = RagflowDatasetClient()
    await dataset_client.list_datasets(page=2, page_size=10, keyword="demo")

    assert captured_call == {
        "method": "GET",
        "path": "/api/v1/datasets",
        "params": {
            "page": 2,
            "page_size": 10,
            "keyword": "demo",
        },
        "json_body": None,
        "expect_envelope": True,
    }


@pytest.mark.asyncio
async def test_ragflow_document_client_assembles_list_request(monkeypatch: MonkeyPatch) -> None:
    """验证文档客户端会按约定组装文档列表查询参数。"""

    captured_call: dict[str, object] = {}

    async def fake_request(
        self: object,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
        expect_envelope: bool = True,
    ) -> dict[str, object]:
        """记录请求参数并返回模拟数据。"""

        del self
        captured_call.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
                "expect_envelope": expect_envelope,
            }
        )
        return {"docs": []}

    monkeypatch.setattr("app.knowledge.ragflow.client.RagflowClient.request", fake_request)

    document_client = RagflowDocumentClient()
    await document_client.list_documents(dataset_id="dataset-001", page=3, page_size=15)

    assert captured_call == {
        "method": "GET",
        "path": "/api/v1/datasets/dataset-001/documents",
        "params": {
            "page": 3,
            "page_size": 15,
            "keyword": None,
        },
        "json_body": None,
        "expect_envelope": True,
    }


@pytest.mark.asyncio
async def test_ragflow_retrieval_client_assembles_request(monkeypatch: MonkeyPatch) -> None:
    """验证检索客户端会按约定组装 retrieval 请求体。"""

    captured_call: dict[str, object] = {}

    async def fake_request(
        self: object,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
        expect_envelope: bool = True,
    ) -> dict[str, object]:
        """记录请求参数并返回模拟数据。"""

        del self
        captured_call.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
                "expect_envelope": expect_envelope,
            }
        )
        return {"chunks": []}

    monkeypatch.setattr("app.knowledge.ragflow.client.RagflowClient.request", fake_request)

    retrieval_client = RagflowRetrievalClient()
    await retrieval_client.retrieve_chunks(
        query="西湖在哪里",
        dataset_ids=["dataset-001", "dataset-002"],
        top_k=3,
    )

    assert captured_call == {
        "method": "POST",
        "path": "/api/v1/retrieval",
        "params": None,
        "json_body": {
            "question": "西湖在哪里",
            "dataset_ids": ["dataset-001", "dataset-002"],
            "top_k": 3,
            "page_size": 3,
        },
        "expect_envelope": True,
    }


@pytest.mark.asyncio
async def test_ragflow_chat_client_assembles_request(monkeypatch: MonkeyPatch) -> None:
    """验证聊天客户端会按约定组装 chat completion 请求。"""

    captured_call: dict[str, object] = {}

    async def fake_request(
        self: object,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
        expect_envelope: bool = True,
    ) -> dict[str, object]:
        """记录请求参数并返回模拟数据。"""

        del self
        captured_call.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
                "expect_envelope": expect_envelope,
            }
        )
        return {"id": "chatcmpl-demo"}

    monkeypatch.setattr("app.knowledge.ragflow.client.RagflowClient.request", fake_request)

    chat_client = RagflowChatClient()
    await chat_client.create_chat_completion(
        chat_id="chat-001",
        request_payload={"question": "你好", "stream": False},
    )

    assert captured_call == {
        "method": "POST",
        "path": "/api/v1/chats/chat-001/completions",
        "params": None,
        "json_body": {"question": "你好", "stream": False},
        "expect_envelope": False,
    }
