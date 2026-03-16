"""知识库节点模块。

负责在命中知识库路由时调用 RAGFlow 检索，并把结果整理成可注入模型的上下文。
当前阶段只做检索增强，不负责复杂引用标注和多知识源编排。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import AgentState
from app.knowledge.service import KnowledgeService
from app.schemas.knowledge import KnowledgeSearchResult

KNOWLEDGE_PROMPT_PREFIX = (
    "以下是知识库检索结果，请优先基于这些内容回答用户问题；如果资料不足请明确说明："
)


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
        if not knowledge_results:
            return {
                "knowledge_results": [],
                "knowledge_context": None,
            }

        return {
            "knowledge_results": knowledge_results,
            "knowledge_context": self._build_knowledge_context(knowledge_results),
        }

    @staticmethod
    def _normalize_query(raw_message: str) -> str:
        """清理知识库路由前缀，保留真正的用户问题。"""

        normalized_query = raw_message.strip()
        if normalized_query.startswith("知识库:"):
            normalized_query = normalized_query.removeprefix("知识库:").strip()
        return normalized_query.replace("#knowledge", "").strip()

    @staticmethod
    def _build_knowledge_context(knowledge_results: list[KnowledgeSearchResult]) -> str:
        """把检索结果拼装成适合注入 system 消息的知识上下文。"""

        context_lines = [KNOWLEDGE_PROMPT_PREFIX]
        for index, knowledge_result in enumerate(knowledge_results, start=1):
            context_lines.append(
                f"[{index}] score={knowledge_result.score:.4f} "
                f"source={knowledge_result.source or knowledge_result.document_id}"
            )
            context_lines.append(knowledge_result.content)
        return "\n".join(context_lines)
