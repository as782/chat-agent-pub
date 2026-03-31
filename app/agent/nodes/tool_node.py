"""工具节点模块。
负责在命中工具或 MCP 路由时执行模型补全、工具调用和二次补全循环。
当前阶段支持内置工具与 MCP 远端工具混合执行，不负责更复杂的跨服务规划。
"""

from json import dumps
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode as PrebuiltToolNode
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.nodes.answer_node import AnswerNode
from app.agent.prompts import UPSTREAM_SERVICE_ERROR_REPLY
from app.agent.state import (
    AgentState,
    ExecutorResult,
    get_execution_step,
    merge_step_result,
)
from app.clients.llm_client import (
    LlmBindableTool,
    LlmClient,
    LlmInputMessage,
    LlmToolCall,
)
from app.core.exceptions import AppException, UpstreamServiceException
from app.mcp.manager import McpManager
from app.mcp.models import McpRuntimeTool
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall, ToolRegistry, tool_to_langchain_format

MAX_TOOL_CALL_ROUNDS = 10


class ToolExecutionUpstreamException(UpstreamServiceException):
    """工具执行阶段的上游接口异常。"""


class ToolNode:
    """LangGraph 工具节点。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        answer_node: AnswerNode,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_manager: McpManager | None = None,
    ) -> None:
        self._answer_node = answer_node
        self._llm_client = llm_client or LlmClient()
        self._tool_registry = tool_registry or ToolRegistry()
        self._mcp_manager = mcp_manager or McpManager()
        self._message_repository = MessageRepository(db_session)
        self._builtin_tool_node = PrebuiltToolNode(
            [tool_to_langchain_format(tool) for tool in self._tool_registry.get_tools()]
        )

    async def run(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, object]:
        """执行工具节点主逻辑。"""

        prepared_context_state = await self._answer_node.prepare_context_state(state)
        prepared_context = prepared_context_state["prepared_context"]
        execution_request = self._answer_node.build_execution_request_from_state(state)
        runtime_mcp_tools = self._extract_runtime_mcp_tools(state)
        available_tools = self._build_available_tools(
            route=str(state.get("route", "answer")),
            requested_tool_names=execution_request.requested_tool_names,
            runtime_mcp_tools=runtime_mcp_tools,
        )
        try:
            completion_result, executed_tool_calls = await self._execute_tool_completion_loop(
                messages=prepared_context.messages,
                model_name=execution_request.model_name,
                session_id=str(state["session_id"]),
                available_tools=available_tools,
                tool_choice=execution_request.tool_choice,
                enable_thinking=execution_request.enable_thinking,
                runtime_mcp_tools=runtime_mcp_tools,
                config=config,
            )
        except ToolExecutionUpstreamException as exception:
            raise UpstreamServiceException(
                UPSTREAM_SERVICE_ERROR_REPLY,
                error_code=exception.error_code,
                status_code=exception.status_code,
                details=exception.details,
            ) from exception
        return {
            **prepared_context_state,
            "tool_completion_result": completion_result,
            "executed_tool_calls": executed_tool_calls,
            **self._build_tool_step_result(
                state=state,
                completion_result=completion_result,
                executed_tool_calls=executed_tool_calls,
            ),
        }

    def build_available_tools(
        self,
        *,
        route: str,
        requested_tool_names: list[str] | None,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[LlmBindableTool]:
        """构造当前轮次可绑定给模型的工具集合。"""

        return self._build_available_tools(
            route=route,
            requested_tool_names=requested_tool_names,
            runtime_mcp_tools=runtime_mcp_tools,
        )

    def build_step_result_update(
        self,
        *,
        state: AgentState,
        completion_result: AIMessage,
        executed_tool_calls: list[ExecutedToolCall],
    ) -> dict[str, object]:
        """瀵瑰鏆撮湶宸ュ叿姝ラ缁撴灉鍚堝苟锛屼緵娴佸紡璺緞鍦ㄥ唴閮ㄥ惊鐜悗閲嶆柊璋冨害銆?"""

        return self._build_tool_step_result(
            state=state,
            completion_result=completion_result,
            executed_tool_calls=executed_tool_calls,
        )

    async def execute_requested_tools_and_persist(
        self,
        *,
        session_id: str,
        completion_result: AIMessage,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[ExecutedToolCall]:
        """执行模型请求的工具，并将 tool 消息持久化。"""

        executed_tool_calls = await self._execute_requested_tools(
            completion_result=completion_result,
            runtime_mcp_tools=runtime_mcp_tools,
        )
        await self._persist_tool_messages(
            session_id=session_id,
            executed_tool_calls=executed_tool_calls,
        )
        return executed_tool_calls

    async def _execute_tool_completion_loop(
        self,
        *,
        messages: list[LlmInputMessage],
        model_name: str | None,
        session_id: str,
        available_tools: list[LlmBindableTool],
        tool_choice: str | dict[str, object] | None,
        enable_thinking: bool | None,
        runtime_mcp_tools: list[McpRuntimeTool],
        config: RunnableConfig | None = None,
    ) -> tuple[AIMessage, list[ExecutedToolCall]]:
        """执行带工具的模型补全循环。"""

        conversation_messages = list(messages)
        normalized_tool_choice = tool_choice or "auto"
        executed_tool_calls: list[ExecutedToolCall] = []

        runnable = self._llm_client.create_runnable(
            model_name=model_name,
            tools=available_tools,
            tool_choice=normalized_tool_choice,
            enable_thinking=enable_thinking,
            is_stream=True,
        )
        for tool_round in range(MAX_TOOL_CALL_ROUNDS):
            llm_messages = self._llm_client._build_langchain_messages(conversation_messages)
            completion_result: AIMessageChunk | None = None
            async for chunk in runnable.astream(llm_messages, config=config):
                if completion_result is None:
                    completion_result = chunk
                else:
                    completion_result = completion_result + chunk

            if completion_result is None:
                raise AppException(
                    "大模型未返回有效响应。",
                    error_code="invalid_llm_response",
                )

            if not completion_result.tool_calls:
                return completion_result, executed_tool_calls

            await self._answer_node.persist_assistant_tool_calls(
                session_id=session_id,
                completion_result=completion_result,
            )

            content = ""
            if isinstance(completion_result.content, str):
                content = completion_result.content
            elif isinstance(completion_result.content, list):
                for part in completion_result.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content += part.get("text", "")

            requested_tool_calls = LlmClient.extract_llm_tool_calls(completion_result)
            conversation_messages.append(
                LlmInputMessage(
                    role="assistant",
                    content=content,
                    tool_calls=requested_tool_calls,
                )
            )
            try:
                current_tool_results = await self._execute_requested_tools(
                    completion_result=completion_result,
                    runtime_mcp_tools=runtime_mcp_tools,
                )
            except UpstreamServiceException as exception:
                raise ToolExecutionUpstreamException(
                    exception.message,
                    error_code=exception.error_code,
                    status_code=exception.status_code,
                    details=exception.details,
                ) from exception
            await self._persist_tool_messages(
                session_id=session_id,
                executed_tool_calls=current_tool_results,
            )
            for tool_result in current_tool_results:
                executed_tool_calls.append(tool_result)
                conversation_messages.append(
                    LlmInputMessage(
                        role="tool",
                        content=tool_result.output,
                        tool_call_id=tool_result.tool_call_id,
                    )
                )
            normalized_tool_choice = "auto"
            if tool_round + 1 >= MAX_TOOL_CALL_ROUNDS:
                raise AppException(
                    "工具调用轮次超过当前上限。",
                    error_code="tool_call_limit_exceeded",
                )

        raise AppException(
            "模型未返回最终回答。",
            error_code="invalid_llm_response",
        )

    def _build_available_tools(
        self,
        *,
        route: str,
        requested_tool_names: list[str] | None,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[LlmBindableTool]:
        """根据路由和状态构造当前轮次允许使用的工具集合。"""

        builtin_tools = (
            []
            if route in {"mcp", "route"}
            else self._tool_registry.get_tools(requested_tool_names)
            if requested_tool_names is not None
            else []
        )
        mcp_tools = [runtime_tool.to_openai_tool() for runtime_tool in runtime_mcp_tools]
        available_tools = [*builtin_tools, *mcp_tools]
        if not available_tools:
            raise AppException(
                "当前请求没有可执行的工具。",
                error_code="no_available_tools",
            )
        return available_tools

    def _build_tool_step_result(
        self,
        *,
        state: AgentState,
        completion_result: AIMessage,
        executed_tool_calls: list[ExecutedToolCall],
    ) -> dict[str, object]:
        """把当前轮次的工具执行结果合并进统一 step_results。"""

        current_step = get_execution_step(
            state,
            step_id=(
                str(state["current_step_id"]) if state.get("current_step_id") is not None else None
            ),
        )
        if current_step is None:
            return {}

        executor_result = ExecutorResult(
            step_id=current_step.step_id,
            executor=current_step.executor,
            is_success=True,
            raw_result={
                "tool_calls": [
                    {
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "output": tool_call.output,
                    }
                    for tool_call in executed_tool_calls
                ],
                "finish_reason": (completion_result.response_metadata or {}).get("finish_reason"),
            },
            normalized_result={
                "executed_tool_count": len(executed_tool_calls),
                "tool_names": [tool_call.tool_name for tool_call in executed_tool_calls],
                "finish_reason": (completion_result.response_metadata or {}).get("finish_reason"),
            },
            summary=(
                f"已完成 {len(executed_tool_calls)} 次工具调用并获得最终结果。"
                if executed_tool_calls
                else "当前步骤未触发工具调用，已直接得到结果。"
            ),
            sources=[tool_call.tool_name for tool_call in executed_tool_calls],
        )
        return merge_step_result(state, result=executor_result)


    async def _execute_requested_tools(
        self,
        *,
        completion_result: AIMessage,
        runtime_mcp_tools: list[McpRuntimeTool],
    ) -> list[ExecutedToolCall]:
        """执行模型本轮返回的全部工具调用。"""

        mcp_tool_map = {
            runtime_tool.registered_name: runtime_tool for runtime_tool in runtime_mcp_tools
        }
        builtin_tool_calls: list[LlmToolCall] = []
        executed_tool_calls: list[ExecutedToolCall] = []

        requested_tool_calls = LlmClient.extract_llm_tool_calls(completion_result)
        for tool_call in requested_tool_calls:
            runtime_mcp_tool = mcp_tool_map.get(tool_call.tool_name)
            if runtime_mcp_tool is None:
                builtin_tool_calls.append(tool_call)
                continue

            tool_response = await self._mcp_manager.call_tool(
                server_name=runtime_mcp_tool.server_name,
                tool_name=runtime_mcp_tool.remote_tool_name,
                arguments=tool_call.arguments,
            )
            normalized_output = tool_response.output_text
            if not normalized_output and tool_response.structured_content is not None:
                normalized_output = dumps(tool_response.structured_content, ensure_ascii=False)
            if not normalized_output:
                normalized_output = "MCP 工具未返回文本结果。"
            executed_tool_calls.append(
                ExecutedToolCall(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=runtime_mcp_tool.registered_name,
                    arguments=tool_call.arguments,
                    output=normalized_output,
                )
            )

        if builtin_tool_calls:
            executed_tool_calls.extend(
                await self._execute_builtin_tool_calls_with_prebuilt_tool_node(
                    completion_result=completion_result,
                    builtin_tool_calls=builtin_tool_calls,
                )
            )

        return executed_tool_calls

    async def _execute_builtin_tool_calls_with_prebuilt_tool_node(
        self,
        *,
        completion_result: AIMessage,
        builtin_tool_calls: list[LlmToolCall],
    ) -> list[ExecutedToolCall]:
        """Use langgraph.prebuilt.ToolNode to execute registered builtin tools."""

        builtin_tool_call_payloads = [
            {
                "id": tool_call.tool_call_id,
                "name": tool_call.tool_name,
                "args": tool_call.arguments,
                "type": "tool_call",
            }
            for tool_call in builtin_tool_calls
        ]
        tool_call_argument_map = {
            tool_call.tool_call_id: tool_call.arguments for tool_call in builtin_tool_calls
        }
        filtered_message = AIMessage(
            content=self._extract_ai_message_text(completion_result),
            tool_calls=builtin_tool_call_payloads,
            response_metadata=completion_result.response_metadata,
        )
        tool_result_state = await self._builtin_tool_node.ainvoke({"messages": [filtered_message]})
        tool_messages = tool_result_state.get("messages", [])

        executed_tool_calls: list[ExecutedToolCall] = []
        for tool_message in tool_messages:
            if not isinstance(tool_message, ToolMessage):
                continue
            executed_tool_calls.append(
                ExecutedToolCall(
                    tool_call_id=str(tool_message.tool_call_id),
                    tool_name=str(tool_message.name or ""),
                    arguments=tool_call_argument_map.get(str(tool_message.tool_call_id), {}),
                    output=self._normalize_tool_message_content(tool_message),
                )
            )
        return executed_tool_calls

    async def _persist_tool_messages(
        self,
        *,
        session_id: str,
        executed_tool_calls: list[ExecutedToolCall],
    ) -> None:
        """把工具执行结果写入消息历史。"""

        for tool_result in executed_tool_calls:
            await self._message_repository.create(
                message_id=uuid4().hex,
                session_id=session_id,
                role="tool",
                content=tool_result.output,
                message_metadata={
                    "tool_call_id": tool_result.tool_call_id,
                    "tool_name": tool_result.tool_name,
                    "arguments": tool_result.arguments,
                },
            )

    @staticmethod
    def extract_runtime_mcp_tools(state: AgentState) -> list[McpRuntimeTool]:
        """从状态中提取当前轮次的 MCP 运行时工具。"""

        raw_runtime_mcp_tools = state.get("mcp_tools", [])
        # LOGGER.info(f"DEBUG: raw_runtime_mcp_tools type: {type(raw_runtime_mcp_tools)}")
        if not isinstance(raw_runtime_mcp_tools, list):
            return []

        result = []
        for rt in raw_runtime_mcp_tools:
            # LOGGER.info(f"DEBUG: tool type: {type(rt)} is_mcp: {isinstance(rt, McpRuntimeTool)}")
            if isinstance(rt, McpRuntimeTool):
                result.append(rt)
            elif isinstance(rt, dict):
                # 兼容字典格式（可能来自序列化）
                try:
                    result.append(McpRuntimeTool(**rt))
                except Exception:
                    pass
        return result

    @staticmethod
    def _extract_runtime_mcp_tools(state: AgentState) -> list[McpRuntimeTool]:
        """兼容旧调用入口。"""

        return ToolNode.extract_runtime_mcp_tools(state)

    @staticmethod
    def _extract_ai_message_text(message: AIMessage) -> str:
        """Normalize AIMessage content into plain text for persistence and replay."""

        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            text_parts: list[str] = []
            for part in message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            return "".join(text_parts)
        return str(message.content)

    @staticmethod
    def _normalize_tool_message_content(message: ToolMessage) -> str:
        """Normalize ToolMessage content into a stable string payload."""

        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            text_parts: list[str] = []
            for part in message.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    else:
                        text_parts.append(dumps(part, ensure_ascii=False))
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return str(message.content)
