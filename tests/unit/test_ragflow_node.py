"""RAGFlow 节点单元测试。"""

from __future__ import annotations

import pytest

from app.agent.nodes.ragflow_node import RagflowNode
from app.agent.state import ResolvedArguments
from app.schemas.knowledge import KnowledgeSearchResult


class _FakeKnowledgeService:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.top_ks: list[int] = []

    async def retrieve_for_agent(
        self,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        self.queries.append(query)
        self.top_ks.append(top_k)

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


@pytest.mark.asyncio
async def test_ragflow_node_expands_scope_check_queries_and_uses_higher_top_k() -> None:
    fake_knowledge_service = _FakeKnowledgeService()
    node = RagflowNode(db_session=None, knowledge_service=fake_knowledge_service)

    result = await node.run(
        {
            "latest_user_message": "鲜花可以走绿通吗",
            "step_arguments": {
                "rag_1": ResolvedArguments(
                    category="policy",
                    arguments={
                        "query": "鲜花可以走绿通吗",
                        "keywords": ["鲜花", "鲜花 绿通", "鲜花 鲜活农产品", "鲜花 鲜活农产品目录"],
                        "query_type": "policy_scope_check",
                        "subject": "鲜花",
                    },
                )
            },
        }
    )

    assert fake_knowledge_service.queries == [
        "鲜花可以走绿通吗",
        "鲜花 是否属于鲜活农产品品种目录",
        "鲜花 是否属于鲜活农产品范围",
        "鲜花 是否适用绿色通道政策",
        "鲜花 不属于 鲜活农产品 范围",
        "鲜花",
        "鲜花 绿通",
        "鲜花 鲜活农产品",
        "鲜花 鲜活农产品目录",
    ]
    assert fake_knowledge_service.top_ks == [6] * 9
    assert result["step_results"]["rag_1"].normalized_result["query_type"] == "policy_scope_check"


def test_ragflow_node_reranks_explicit_scope_evidence_ahead_of_generic_admin_text() -> None:
    generic_admin = KnowledgeSearchResult(
        document_id="doc-admin",
        chunk_id="chunk-admin",
        score=0.91,
        content="收费站查验时需核验预约信息、公示牌和投诉电话。",
        source="admin",
    )
    explicit_negative = KnowledgeSearchResult(
        document_id="doc-negative",
        chunk_id="chunk-negative",
        score=0.72,
        content="花、草、苗木、粮食等不属于鲜活农产品范围，不适用绿色通道运输政策。",
        source="policy",
    )

    reranked = RagflowNode._rerank_knowledge_results(
        [generic_admin, explicit_negative],
        step_arguments=ResolvedArguments(
            category="policy",
            arguments={"query_type": "policy_scope_check", "subject": "鲜花"},
        ),
    )

    assert reranked[0].document_id == "doc-negative"


def test_ragflow_node_builds_scope_context_with_evidence_guardrail() -> None:
    explicit_negative = KnowledgeSearchResult(
        document_id="doc-negative",
        chunk_id="chunk-negative",
        score=0.72,
        content="鲜花不属于鲜活农产品范围，不适用绿色通道运输政策。",
        source="policy",
    )

    context = RagflowNode._build_knowledge_context(
        [explicit_negative],
        step_arguments=ResolvedArguments(
            category="policy",
            arguments={"query_type": "policy_scope_check", "subject": "鲜花"},
        ),
    )

    assert "当前问题属于“对象是否属于政策适用范围”的判断题" in context
    assert "若没有检索到主体的明确归属证据，不要直接下确定结论" in context
    assert "证据摘要：" in context
    assert "鲜花不属于鲜活农产品范围" in context
