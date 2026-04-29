"""知识库节点模块。
负责在命中知识库路由时调用 RAGFlow 检索，并把结果整理成可注入模型的上下文。
当前阶段只做检索增强，不负责复杂引用标注和多知识源编排。
"""

from __future__ import annotations

from collections import OrderedDict
from re import search

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompts import KNOWLEDGE_CONTEXT_PROMPT_PREFIX, UPSTREAM_SERVICE_ERROR_REPLY
from app.agent.state import (
    AgentState,
    ExecutorResult,
    ResolvedArguments,
    merge_step_result,
    resolve_active_execution_step_id,
    resolve_step_arguments,
)
from app.core.exceptions import UpstreamServiceException
from app.knowledge.service import KnowledgeService
from app.schemas.knowledge import KnowledgeSearchResult


class RagflowNode:
    """LangGraph 知识库节点。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        knowledge_service: KnowledgeService | None = None,
    ) -> None:
        self._knowledge_service = knowledge_service or KnowledgeService(db_session)

    async def run(self, state: AgentState) -> dict[str, object]:
        """执行知识检索，并返回注入回答节点所需的知识上下文。"""

        step_id = resolve_active_execution_step_id(
            state,
            executor="rag",
            default_step_id="rag_1",
        )
        step_arguments = resolve_step_arguments(state, step_id=step_id, executor="rag")
        retrieval_queries = self._resolve_queries(state, step_arguments)
        normalized_query = retrieval_queries[0]
        try:
            knowledge_results = await self._retrieve_knowledge_results(
                retrieval_queries,
                step_arguments=step_arguments,
            )
        except UpstreamServiceException as exception:
            raise UpstreamServiceException(
                UPSTREAM_SERVICE_ERROR_REPLY,
                error_code=exception.error_code,
                status_code=exception.status_code,
                details=exception.details,
            ) from exception
        if not knowledge_results:
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="rag",
                is_success=True,
                raw_result={
                    "query_arguments": self._serialize_step_arguments(step_arguments),
                    "retrieval_queries": retrieval_queries,
                    "results": [],
                },
                normalized_result={
                    "query": normalized_query,
                    "queries": retrieval_queries,
                    "keywords": self._resolve_keywords(step_arguments),
                    "query_type": self._resolve_query_type(step_arguments),
                    "result_count": 0,
                },
                summary="知识检索未命中结果。",
            )
            return {
                "knowledge_results": [],
                "knowledge_context": None,
                **merge_step_result(state, result=executor_result),
            }

        executor_result = ExecutorResult(
            step_id=step_id,
            executor="rag",
            is_success=True,
            raw_result={
                "query_arguments": self._serialize_step_arguments(step_arguments),
                "retrieval_queries": retrieval_queries,
                "results": [
                    knowledge_result.model_dump(mode="json")
                    for knowledge_result in knowledge_results
                ]
            },
            normalized_result={
                "query": normalized_query,
                "queries": retrieval_queries,
                "keywords": self._resolve_keywords(step_arguments),
                "query_type": self._resolve_query_type(step_arguments),
                "result_count": len(knowledge_results),
                "sources": [
                    knowledge_result.source or knowledge_result.document_id
                    for knowledge_result in knowledge_results
                ],
            },
            summary=f"知识检索命中 {len(knowledge_results)} 条结果。",
            sources=[
                knowledge_result.source or knowledge_result.document_id
                for knowledge_result in knowledge_results
            ],
        )
        return {
            "knowledge_results": knowledge_results,
            "knowledge_context": self._build_knowledge_context(
                knowledge_results,
                step_arguments=step_arguments,
            ),
            **merge_step_result(state, result=executor_result),
        }

    @staticmethod
    def _resolve_query(
        state: AgentState,
        step_arguments: ResolvedArguments | None,
    ) -> str:
        """优先使用当前 rag 步骤参数中的 query，回退到原消息归一化。"""

        if isinstance(step_arguments, ResolvedArguments):
            query = str(step_arguments.arguments.get("query") or "").strip()
            if query:
                return query
        return RagflowNode._normalize_query(str(state.get("latest_user_message", "")))

    @classmethod
    def _resolve_queries(
        cls,
        state: AgentState,
        step_arguments: ResolvedArguments | None,
    ) -> list[str]:
        queries = [cls._resolve_query(state, step_arguments)]
        query_type = cls._resolve_query_type(step_arguments)
        subject = cls._resolve_subject(step_arguments)
        if query_type == "policy_scope_check" and subject is not None:
            queries.extend(cls._build_policy_scope_queries(subject=subject))
        queries.extend(cls._resolve_keywords(step_arguments))
        return cls._deduplicate_queries(queries)

    async def _retrieve_knowledge_results(
        self,
        retrieval_queries: list[str],
        *,
        step_arguments: ResolvedArguments | None,
    ) -> list[KnowledgeSearchResult]:
        merged_results: OrderedDict[tuple[str, str, str], KnowledgeSearchResult] = OrderedDict()
        query_type = self._resolve_query_type(step_arguments)
        per_query_top_k = 6 if query_type == "policy_scope_check" else 4

        for query in retrieval_queries:
            current_results = await self._knowledge_service.retrieve_for_agent(
                query=query,
                top_k=per_query_top_k,
            )
            for result in current_results:
                result_key = (
                    result.document_id,
                    result.chunk_id,
                    result.source or "",
                )
                existing_result = merged_results.get(result_key)
                if existing_result is None or result.score > existing_result.score:
                    merged_results[result_key] = result

        return self._rerank_knowledge_results(
            list(merged_results.values()),
            step_arguments=step_arguments,
        )

    @staticmethod
    def _serialize_step_arguments(
        step_arguments: ResolvedArguments | None,
    ) -> dict[str, object]:
        if not isinstance(step_arguments, ResolvedArguments):
            return {}
        return dict(step_arguments.arguments)

    @staticmethod
    def _resolve_keywords(step_arguments: ResolvedArguments | None) -> list[str]:
        if not isinstance(step_arguments, ResolvedArguments):
            return []

        raw_keywords = step_arguments.arguments.get("keywords")
        if not isinstance(raw_keywords, list):
            return []

        return [
            str(keyword).strip()
            for keyword in raw_keywords
            if str(keyword).strip()
        ]

    @staticmethod
    def _resolve_query_type(step_arguments: ResolvedArguments | None) -> str | None:
        if not isinstance(step_arguments, ResolvedArguments):
            return None

        query_type = step_arguments.arguments.get("query_type")
        if isinstance(query_type, str):
            return query_type.strip() or None
        return None

    @staticmethod
    def _resolve_subject(step_arguments: ResolvedArguments | None) -> str | None:
        if not isinstance(step_arguments, ResolvedArguments):
            return None

        subject = step_arguments.arguments.get("subject")
        if isinstance(subject, str):
            return subject.strip() or None
        return None

    @staticmethod
    def _build_policy_scope_queries(*, subject: str) -> list[str]:
        return [
            f"{subject} 是否属于鲜活农产品品种目录",
            f"{subject} 是否属于鲜活农产品范围",
            f"{subject} 是否适用绿色通道政策",
            f"{subject} 不属于 鲜活农产品 范围",
        ]

    @classmethod
    def _rerank_knowledge_results(
        cls,
        knowledge_results: list[KnowledgeSearchResult],
        *,
        step_arguments: ResolvedArguments | None,
    ) -> list[KnowledgeSearchResult]:
        query_type = cls._resolve_query_type(step_arguments)
        subject = cls._resolve_subject(step_arguments)
        if query_type != "policy_scope_check" or subject is None:
            return sorted(knowledge_results, key=lambda item: item.score, reverse=True)

        reranked_results = sorted(
            knowledge_results,
            key=lambda item: cls._score_policy_scope_result(item, subject=subject),
            reverse=True,
        )
        return reranked_results[:8]

    @staticmethod
    def _score_policy_scope_result(
        knowledge_result: KnowledgeSearchResult,
        *,
        subject: str,
    ) -> float:
        content = knowledge_result.content
        normalized_content = content.replace(" ", "")
        normalized_subject = subject.replace(" ", "")
        score = knowledge_result.score * 100

        if normalized_subject and normalized_subject in normalized_content:
            score += 35
        if any(token in content for token in ("属于", "不属于", "适用", "不适用", "包含", "不包含", "范围", "目录")):
            score += 20
        if any(token in content for token in ("问：", "答：", "是否", "算不算")):
            score += 10
        if any(token in content for token in ("不属于", "不适用", "不包含")):
            score += 12
        if any(token in content for token in ("目录", "品种目录", "鲜活农产品范围")):
            score += 8
        if any(token in content for token in ("预约", "收费站", "投诉电话", "公示牌", "查验", "预约平台")):
            score -= 10
        if search(r"花、草、苗木|苗木、粮食|深加工产品", content) is not None:
            score += 12
        return score

    @staticmethod
    def _deduplicate_queries(queries: list[str]) -> list[str]:
        deduplicated_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized_query = query.strip()
            if not normalized_query or normalized_query in seen_queries:
                continue
            deduplicated_queries.append(normalized_query)
            seen_queries.add(normalized_query)
        return deduplicated_queries

    @staticmethod
    def _normalize_query(raw_message: str) -> str:
        """清理知识库路由前缀，保留真正的用户问题。"""

        normalized_query = raw_message.strip()
        for prefix in ("知识库:", "knowledge:", "konwledge:"):
            if normalized_query.lower().startswith(prefix.lower()):
                normalized_query = normalized_query[len(prefix) :].strip()
                break
        return normalized_query.replace("#knowledge", "").strip()

    @classmethod
    def _build_knowledge_context(
        cls,
        knowledge_results: list[KnowledgeSearchResult],
        *,
        step_arguments: ResolvedArguments | None,
    ) -> str:
        """把检索结果拼装成适合注入 system 消息的知识上下文。"""

        context_lines = [KNOWLEDGE_CONTEXT_PROMPT_PREFIX]
        query_type = cls._resolve_query_type(step_arguments)
        subject = cls._resolve_subject(step_arguments)
        if query_type == "policy_scope_check" and subject is not None:
            context_lines.append(
                (
                    "当前问题属于“对象是否属于政策适用范围”的判断题。"
                    "请优先依据同时包含主体和“属于/不属于/适用/不适用/目录/范围”等表述的证据作答；"
                    "若没有检索到主体的明确归属证据，不要直接下确定结论。"
                )
            )
            evidence_summary = cls._build_policy_scope_evidence_summary(
                knowledge_results,
                subject=subject,
            )
            if evidence_summary is not None:
                context_lines.append(evidence_summary)
        for index, knowledge_result in enumerate(knowledge_results, start=1):
            context_lines.append(
                f"[{index}] score={knowledge_result.score:.4f} "
                f"source={knowledge_result.source or knowledge_result.document_id}"
            )
            context_lines.append(knowledge_result.content)
        return "\n".join(context_lines)

    @staticmethod
    def _build_policy_scope_evidence_summary(
        knowledge_results: list[KnowledgeSearchResult],
        *,
        subject: str,
    ) -> str | None:
        matched_fragments: list[str] = []
        normalized_subject = subject.replace(" ", "")
        for knowledge_result in knowledge_results[:5]:
            content = knowledge_result.content.replace(" ", "")
            if normalized_subject and normalized_subject not in content:
                continue
            if any(token in content for token in ("不属于", "不适用", "不包含", "属于", "适用", "包含")):
                fragment = knowledge_result.content.strip().replace("\n", " ")
                matched_fragments.append(fragment[:180])
            if len(matched_fragments) >= 2:
                break
        if not matched_fragments:
            return None
        summary_lines = ["证据摘要："]
        summary_lines.extend(f"- {fragment}" for fragment in matched_fragments)
        return "\n".join(summary_lines)
