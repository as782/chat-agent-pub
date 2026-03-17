"""Agent 对话图模块。
负责组装 LangGraph 状态图，并暴露单轮运行与流式上下文准备入口。
当前阶段先把 planner 节点接入图中，但仍保持既有真实路由行为不变。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.nodes.answer_node import AnswerNode
from app.agent.nodes.argument_node import ArgumentNode
from app.agent.nodes.mcp_node import McpNode
from app.agent.nodes.memory_node import MemoryNode
from app.agent.nodes.planner_node import PlannerNode
from app.agent.nodes.ragflow_node import RagflowNode
from app.agent.nodes.report_node import ReportNode
from app.agent.nodes.router_node import RouterNode
from app.agent.nodes.tool_node import ToolNode
from app.agent.nodes.traffic_node import TrafficNode
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
        self._planner_node = PlannerNode()
        self._argument_node = ArgumentNode()
        self._router_node = RouterNode()
        self._traffic_node = TrafficNode()
        self._report_node = ReportNode()
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

    async def prepare_stream_state(
        self,
        execution_request: ChatExecutionRequest,
    ) -> AgentState:
        """为流式路径准备与主图一致的预处理状态。"""

        initial_state = self._build_initial_state(execution_request)
        merged_state = await self._prepare_route_state(initial_state)
        context_state = await self._answer_node.prepare_context_state(merged_state)
        return {**merged_state, **context_state}

    async def prepare_stream_context(
        self,
        execution_request: ChatExecutionRequest,
    ) -> tuple[str, PreparedContext]:
        """兼容旧流式调用入口。"""

        prepared_state = await self.prepare_stream_state(execution_request)
        prepared_context = prepared_state["prepared_context"]
        return str(prepared_state["route"]), prepared_context

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

    def get_tool_node(self) -> ToolNode:
        """暴露工具节点，供流式路径复用工具执行逻辑。"""

        return self._tool_node

    def _build_graph(self) -> Any:
        """组装 LangGraph 状态图。"""

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("planner_node", self._planner_node.run)
        graph_builder.add_node("argument_node", self._argument_node.run)
        graph_builder.add_node("router_node", self._router_node.run)
        graph_builder.add_node("tool_node", self._tool_node.run)
        graph_builder.add_node("ragflow_node", self._ragflow_node.run)
        graph_builder.add_node("mcp_node", self._mcp_node.run)
        graph_builder.add_node("traffic_node", self._traffic_node.run)
        graph_builder.add_node("report_node", self._report_node.run)
        graph_builder.add_node("answer_node", self._answer_node.run)
        graph_builder.add_node("memory_node", self._memory_node.run)

        graph_builder.add_edge(START, "planner_node")
        graph_builder.add_edge("planner_node", "argument_node")
        graph_builder.add_edge("argument_node", "router_node")
        graph_builder.add_conditional_edges(
            "router_node",
            self._resolve_next_node,
            {
                "tool_node": "tool_node",
                "answer_node": "answer_node",
                "ragflow_node": "ragflow_node",
                "mcp_node": "mcp_node",
                "traffic_node": "traffic_node",
                "report_node": "report_node",
            },
        )
        graph_builder.add_edge("tool_node", "memory_node")
        graph_builder.add_edge("ragflow_node", "answer_node")
        graph_builder.add_edge("mcp_node", "tool_node")
        graph_builder.add_edge("traffic_node", "answer_node")
        graph_builder.add_edge("report_node", "answer_node")
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
            "enable_thinking": execution_request.enable_thinking,
        }

    @staticmethod
    def _resolve_next_node(state: AgentState) -> str:
        """根据路由结果选择下一节点。"""

        route = state.get("route", "answer")
        if route == "tool":
            return "tool_node"
        if route == "ragflow":
            return "ragflow_node"
        if route == "mcp":
            return "mcp_node"
        if route == "traffic":
            return "traffic_node"
        if route == "report":
            return "report_node"
        return "answer_node"

    async def _prepare_route_state(self, initial_state: AgentState) -> AgentState:
        """按真实路由顺序执行前置节点，供流式路径复用。"""

        planned_state = await self._planner_node.run(initial_state)
        argument_state = await self._argument_node.run({**initial_state, **planned_state})
        route_state = await self._router_node.run(
            {**initial_state, **planned_state, **argument_state}
        )
        merged_state: AgentState = {
            **initial_state,
            **planned_state,
            **argument_state,
            **route_state,
        }
        if merged_state.get("route") == "ragflow":
            knowledge_state = await self._ragflow_node.run(merged_state)
            merged_state = {**merged_state, **knowledge_state}
        if merged_state.get("route") == "mcp":
            mcp_state = await self._mcp_node.run(merged_state)
            merged_state = {**merged_state, **mcp_state}
        if merged_state.get("route") == "traffic":
            traffic_state = await self._traffic_node.run(merged_state)
            merged_state = {**merged_state, **traffic_state}
        if merged_state.get("route") == "report":
            report_state = await self._report_node.run(merged_state)
            merged_state = {**merged_state, **report_state}
        return merged_state
