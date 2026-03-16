"""Agent 对话图模块。
负责组装 LangGraph 状态图，并暴露单轮运行与流式上下文准备入口。
当前阶段主图只接入路由、回答和记忆节点，不负责知识库与 MCP 独立分支。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.nodes.answer_node import AnswerNode
from app.agent.nodes.mcp_node import McpNode
from app.agent.nodes.memory_node import MemoryNode
from app.agent.nodes.ragflow_node import RagflowNode
from app.agent.nodes.router_node import RouterNode
from app.agent.nodes.tool_node import ToolNode
from app.agent.state import AgentState, ChatExecutionRequest, ChatTurnResult, PreparedContext
from app.clients.llm_client import LlmClient
from app.tools.registry import ToolRegistry


class ConversationGraph:
    """最小可用 LangGraph 对话图。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        shared_llm_client = llm_client or LlmClient()
        shared_tool_registry = tool_registry or ToolRegistry()
        self._router_node = RouterNode()
        self._answer_node = AnswerNode(
            db_session,
            llm_client=shared_llm_client,
            tool_registry=shared_tool_registry,
        )
        self._tool_node = ToolNode(
            db_session,
            answer_node=self._answer_node,
            llm_client=shared_llm_client,
            tool_registry=shared_tool_registry,
        )
        self._ragflow_node = RagflowNode(db_session)
        self._mcp_node = McpNode()
        self._memory_node = MemoryNode(db_session)
        self._compiled_graph = self._build_graph()

    async def run_turn(
        self,
        execution_request: ChatExecutionRequest,
    ) -> tuple[ChatTurnResult, dict[str, object] | None]:
        """运行一轮完整的 LangGraph 对话图。"""

        initial_state = self._build_initial_state(execution_request)
        final_state = await self._compiled_graph.ainvoke(initial_state)
        final_result = final_state.get("final_result")
        if not isinstance(final_result, ChatTurnResult):
            raise RuntimeError("LangGraph 未返回有效的对话结果。")
        checkpoint_payload = final_state.get("checkpoint_payload")
        return (
            final_result,
            checkpoint_payload if isinstance(checkpoint_payload, dict) else None,
        )

    async def prepare_stream_context(
        self,
        execution_request: ChatExecutionRequest,
    ) -> tuple[str, PreparedContext]:
        """为流式路径准备与主图一致的上下文。"""

        initial_state = self._build_initial_state(execution_request)
        merged_state = await self._prepare_route_state(initial_state)
        context_state = await self._answer_node.prepare_context_state(merged_state)
        prepared_context = context_state["prepared_context"]
        return str(merged_state["route"]), prepared_context

    async def refresh_memory(
        self,
        *,
        session_id: str,
        route: str,
    ) -> dict[str, object]:
        """在流式路径结束后刷新会话记忆。"""

        return await self._memory_node.refresh_session_memory(session_id=session_id, route=route)

    async def save_checkpoint(self, checkpoint_payload: dict[str, object] | None) -> None:
        """在事务提交后保存 checkpoint。"""

        await self._memory_node.save_checkpoint(checkpoint_payload)

    def get_answer_node(self) -> AnswerNode:
        """暴露回答节点，供流式路径复用持久化逻辑。"""

        return self._answer_node

    def _build_graph(self) -> Any:
        """组装 LangGraph 状态图。"""

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("router_node", self._router_node.run)
        graph_builder.add_node("tool_node", self._tool_node.run)
        graph_builder.add_node("ragflow_node", self._ragflow_node.run)
        graph_builder.add_node("mcp_node", self._mcp_node.run)
        graph_builder.add_node("answer_node", self._answer_node.run)
        graph_builder.add_node("memory_node", self._memory_node.run)

        graph_builder.add_edge(START, "router_node")
        graph_builder.add_conditional_edges(
            "router_node",
            self._resolve_next_node,
            {
                "tool_node": "tool_node",
                "answer_node": "answer_node",
                "ragflow_node": "ragflow_node",
                "mcp_node": "mcp_node",
            },
        )
        graph_builder.add_edge("tool_node", "memory_node")
        graph_builder.add_edge("ragflow_node", "answer_node")
        graph_builder.add_edge("mcp_node", "tool_node")
        graph_builder.add_edge("answer_node", "memory_node")
        graph_builder.add_edge("memory_node", END)
        return graph_builder.compile()

    @staticmethod
    def _build_initial_state(execution_request: ChatExecutionRequest) -> AgentState:
        """把统一执行请求转换为图初始状态。"""

        if execution_request.session_id is None:
            raise RuntimeError("构建对话图状态前必须先解析会话标识。")

        return {
            "session_id": execution_request.session_id,
            "need_session_memory": execution_request.need_session_memory,
            "user_id": execution_request.user_id,
            "latest_user_message": execution_request.latest_user_message,
            "input_messages": execution_request.input_messages,
            "model_name": execution_request.model_name,
            "requested_tool_names": execution_request.requested_tool_names,
            "tool_choice": execution_request.tool_choice,
        }

    @staticmethod
    def _resolve_next_node(state: AgentState) -> str:
        """根据路由结果选择下一个节点。"""

        route = state.get("route", "answer")
        if route == "tool":
            return "tool_node"
        if route == "ragflow":
            return "ragflow_node"
        if route == "mcp":
            return "mcp_node"
        if route == "answer":
            return "answer_node"
        return "answer_node"

    async def _prepare_route_state(self, initial_state: AgentState) -> AgentState:
        """按真实路由顺序执行前置节点，供流式路径复用。"""

        route_state = await self._router_node.run(initial_state)
        merged_state: AgentState = {**initial_state, **route_state}
        if merged_state.get("route") == "ragflow":
            knowledge_state = await self._ragflow_node.run(merged_state)
            merged_state = {**merged_state, **knowledge_state}
        if merged_state.get("route") == "mcp":
            mcp_state = await self._mcp_node.run(merged_state)
            merged_state = {**merged_state, **mcp_state}
        return merged_state
