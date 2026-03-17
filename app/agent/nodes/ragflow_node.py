"""知识库节点模块。
负责在命中知识库路由时调用 RAGFlow 检索，并把结果整理成可注入模型的上下文。
当前阶段只做检索增强，不负责复杂引用标注和多知识源编排。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompts import KNOWLEDGE_CONTEXT_PROMPT_PREFIX
from app.agent.state import AgentState, ExecutorResult, merge_step_result, resolve_execution_step_id
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

        normalized_query = self._normalize_query(str(state.get("latest_user_message", "")))
        knowledge_results = await self._knowledge_service.retrieve_for_agent(query=normalized_query)
        step_id = resolve_execution_step_id(state, executor="rag", default_step_id="rag_1")
        if not knowledge_results:
            executor_result = ExecutorResult(
                step_id=step_id,
                executor="rag",
                is_success=True,
                raw_result={"results": []},
                normalized_result={"result_count": 0, "query": normalized_query},
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
                "results": [
                    knowledge_result.model_dump(mode="json")
                    for knowledge_result in knowledge_results
                ]
            },
            normalized_result={
                "query": normalized_query,
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
            "knowledge_context": self._build_knowledge_context(knowledge_results),
            **merge_step_result(state, result=executor_result),
        }

    @staticmethod
    def _normalize_query(raw_message: str) -> str:
        """清理知识库路由前缀，保留真正的用户问题。"""

        normalized_query = raw_message.strip()
        for prefix in ("知识库:", "knowledge:", "konwledge:"):
            if normalized_query.lower().startswith(prefix.lower()):
                normalized_query = normalized_query[len(prefix) :].strip()
                break
        return normalized_query.replace("#knowledge", "").strip()

    @staticmethod
    def _build_knowledge_context(knowledge_results: list[KnowledgeSearchResult]) -> str:
        """把检索结果拼装成适合注入 system 消息的知识上下文。"""

        context_lines = [KNOWLEDGE_CONTEXT_PROMPT_PREFIX]
        for index, knowledge_result in enumerate(knowledge_results, start=1):
            context_lines.append(
                f"[{index}] score={knowledge_result.score:.4f} "
                f"source={knowledge_result.source or knowledge_result.document_id}"
            )
            context_lines.append(knowledge_result.content)
        return "\n".join(context_lines)
