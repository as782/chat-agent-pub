"""RAGFlow 节点单元测试。"""

from __future__ import annotations

import pytest

from app.agent.nodes.ragflow_node import RagflowNode
from app.agent.state import ResolvedArguments
from app.schemas.knowledge import KnowledgeSearchResult


class _FakeKnowledgeService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve_for_agent(
        self,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        del top_k
        self.queries.append(query)

        if query == "今天上高速到明天下高速要过路费吗":
            return [
                KnowledgeSearchResult(
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    score=0.62,
                    content="节假日免费以车辆驶离出口收费车道的时间为准。",
                    source="policy-1",
                )
            ]
        if query == "高速过路费":
            return [
                KnowledgeSearchResult(
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    score=0.51,
                    content="节假日免费以车辆驶离出口收费车道的时间为准。",
                    source="policy-1",
                )
            ]
        if query == "跨天":
            return [
                KnowledgeSearchResult(
                    document_id="doc-2",
                    chunk_id="chunk-2",
                    score=0.58,
                    content="跨天通行时，应结合上下高速时间和免费时段判断。",
                    source="policy-2",
                )
            ]
        if query == "收费规则":
            return [
                KnowledgeSearchResult(
                    document_id="doc-3",
                    chunk_id="chunk-3",
                    score=0.55,
                    content="普通收费时段内仍按车型和里程计费。",
                    source="policy-3",
                )
            ]
        return []


@pytest.mark.asyncio
async def test_ragflow_node_uses_query_and_keywords_for_retrieval() -> None:
    fake_knowledge_service = _FakeKnowledgeService()
    node = RagflowNode(db_session=None, knowledge_service=fake_knowledge_service)

    result = await node.run(
        {
            "latest_user_message": "今天上高速到明天下高速要过路费吗",
            "step_arguments": {
                "rag_1": ResolvedArguments(
                    category="policy",
                    arguments={
                        "query": "今天上高速到明天下高速要过路费吗",
                        "keywords": ["高速过路费", "跨天", "收费规则"],
                        "query_type": "policy_interpretation",
                    },
                )
            },
        }
    )

    assert fake_knowledge_service.queries == [
        "今天上高速到明天下高速要过路费吗",
        "高速过路费",
        "跨天",
        "收费规则",
    ]
    assert result["step_results"]["rag_1"].normalized_result["query"] == "今天上高速到明天下高速要过路费吗"
    assert result["step_results"]["rag_1"].normalized_result["queries"] == [
        "今天上高速到明天下高速要过路费吗",
        "高速过路费",
        "跨天",
        "收费规则",
    ]
    assert result["step_results"]["rag_1"].normalized_result["keywords"] == [
        "高速过路费",
        "跨天",
        "收费规则",
    ]
    assert result["step_results"]["rag_1"].normalized_result["query_type"] == "policy_interpretation"
    assert result["step_results"]["rag_1"].normalized_result["result_count"] == 3


@pytest.mark.asyncio
async def test_ragflow_node_deduplicates_queries_and_results() -> None:
    fake_knowledge_service = _FakeKnowledgeService()
    node = RagflowNode(db_session=None, knowledge_service=fake_knowledge_service)

    result = await node.run(
        {
            "latest_user_message": "今天上高速到明天下高速要过路费吗",
            "step_arguments": {
                "rag_1": ResolvedArguments(
                    category="policy",
                    arguments={
                        "query": "今天上高速到明天下高速要过路费吗",
                        "keywords": ["高速过路费", "高速过路费", "跨天"],
                    },
                )
            },
        }
    )

    assert fake_knowledge_service.queries == [
        "今天上高速到明天下高速要过路费吗",
        "高速过路费",
        "跨天",
    ]
    assert result["step_results"]["rag_1"].normalized_result["result_count"] == 2
