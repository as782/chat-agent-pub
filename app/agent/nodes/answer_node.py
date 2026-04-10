"""回答节点模块。

负责构建模型上下文、执行普通回答补全，并持久化本轮输出。
当前阶段不负责工具循环编排，这部分能力由独立的 tool_node 承担。
"""

from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context_builder import ContextBuilder
from app.agent.prompts import (
    GENERAL_ANSWER_PROMPT,
    NETWORK_REPORT_SUMMARY_PROMPT,
    POLICY_SUMMARY_PROMPT,
    ROUTE_SUMMARY_PROMPT,
    SERVICE_SUMMARY_PROMPT,
    TRAFFIC_SUMMARY_PROMPT,
)
from app.agent.state import (
    AgentState,
    ChatExecutionRequest,
    ChatTurnResult,
    ExecutorResult,
    PreparedContext,
    get_execution_step,
)
from app.clients.llm_client import LlmClient, LlmInputMessage
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logger import get_logger
from app.memory.manager import MemoryManager
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall, ToolRegistry

LOGGER = get_logger(__name__)

RECENT_CONTEXT_WINDOW_SIZE = 8
_PROMPT_NAME_BY_CATEGORY = {
    "policy": "POLICY_SUMMARY_PROMPT",
    "route_planning": "ROUTE_SUMMARY_PROMPT",
    "traffic_status": "TRAFFIC_SUMMARY_PROMPT",
    "service_area": "SERVICE_SUMMARY_PROMPT",
    "network_report": "NETWORK_REPORT_SUMMARY_PROMPT",
    "general": "GENERAL_ANSWER_PROMPT",
}

_TRAFFIC_KEYWORDS = (
    "堵不堵",
    "拥堵",
    "路况",
    "堵吗",
    "缓行",
    "施工",
    "事故",
    "封闭",
    "管制",
    "通行",
    "是否畅通",
    "是否拥堵",
    "会不会堵",
)
_SERVICE_KEYWORDS = (
    "服务区",
    "充电桩",
    "充电",
    "加油",
    "休息区",
    "配套",
)
_POLICY_KEYWORDS = (
    "政策",
    "规则",
    "标准",
    "制度",
    "口径",
    "收费",
    "绿通",
    "免费",
)
_ROUTE_KEYWORDS = (
    "怎么走",
    "如何走",
    "路线",
    "路线路",
    "导航",
    "出行方案",
)
_REPORT_KEYWORDS = (
    "报表",
    "汇总",
    "对比",
    "全省",
    "全网",
    "路网",
    "总体情况",
)


class AnswerNode:
    """LangGraph 回答节点。"""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        llm_client: LlmClient | None = None,
        tool_registry: ToolRegistry | None = None,
        context_builder: ContextBuilder | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._llm_client = llm_client or LlmClient()
        self._tool_registry = tool_registry or ToolRegistry()
        self._context_builder = context_builder or ContextBuilder()
        self._memory_manager = memory_manager or MemoryManager(db_session)
        self._message_repository = MessageRepository(db_session)
        self._settings = get_settings()

    async def run(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, object]:
        """执行普通回答节点主逻辑。"""

        prepared_context = state.get("prepared_context")
        if isinstance(prepared_context, PreparedContext):
            prepared_context_state: dict[str, PreparedContext] = {
                "prepared_context": prepared_context
            }
        else:
            prepared_context_state = await self.prepare_context_state(state)
            prepared_context = prepared_context_state["prepared_context"]

        completion_result = state.get("tool_completion_result")
        executed_tool_calls = self._extract_executed_tool_calls(state)
        existing_tool_completion_result = completion_result
        should_reuse_existing_completion = isinstance(
            completion_result,
            AIMessage,
        ) and not self.should_generate_summary(state)

        if not should_reuse_existing_completion:
            execution_request = self.build_execution_request_from_state(state)
            llm_messages = self._llm_client._build_langchain_messages(
                prepared_context.messages,
                model_name=execution_request.model_name,
            )
            runnable = self._llm_client.create_runnable(
                messages=prepared_context.messages,
                model_name=execution_request.model_name,
                api_key=(
                    self._settings.openai_api_key.get_secret_value()
                    if self._settings.openai_api_key
                    else None
                ),
                base_url=self._settings.openai_base_url,
                timeout_seconds=self._settings.openai_timeout_seconds,
                enable_thinking=execution_request.enable_thinking,
                is_stream=True,
            )

            completion_result: AIMessageChunk | None = None
            received_any_chunk = False
            accumulated_content = ""
            accumulated_reasoning = ""
            has_reasoning_field = False
            
            async for chunk in runnable.astream(llm_messages, config=config):
                received_any_chunk = True
                if completion_result is None:
                    completion_result = chunk
                else:
                    completion_result = completion_result + chunk
                
                # 累积内容用于检查
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_content += chunk.content
                if hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
                    if 'reasoning_content' in chunk.additional_kwargs:
                        has_reasoning_field = True
                        reasoning = chunk.additional_kwargs['reasoning_content']
                        if reasoning:  # 只有非空时才累积
                            accumulated_reasoning += reasoning

            # 检查是否有实际内容或reasoning字段
            has_content_or_reasoning = bool(accumulated_content or accumulated_reasoning or has_reasoning_field)

            # 区分 "没收到任何chunk" 和 "只收到thinking chunks的情况"
            if completion_result is None:
                if received_any_chunk:
                    # 收到了 chunks 但都被过滤/为空（可能是纯 thinking 模式）
                    LOGGER.warning(
                        "大模型返回了 chunks 但无法构建 completion_result。"
                        "这可能是因为启用了 thinking 模式且模型只返回了思考内容。"
                    )
                    raise AppException(
                        "大模型返回了响应但无法处理（可能是纯思考内容）。",
                        error_code="invalid_llm_response",
                    )
                else:
                    # 完全没收到任何 chunk
                    raise AppException(
                        "大模型未返回任何响应内容。",
                        error_code="invalid_llm_response",
                    )
            
            # 如果只有 reasoning_content 没有 content，也认为是有效的响应
            if not has_content_or_reasoning and completion_result:
                # 再次检查最终结果（以防万一）
                if hasattr(completion_result, 'additional_kwargs') and completion_result.additional_kwargs:
                    if 'reasoning_content' in completion_result.additional_kwargs and completion_result.additional_kwargs['reasoning_content']:
                        has_content_or_reasoning = True
                
                if not has_content_or_reasoning:
                    LOGGER.warning(
                        "大模型返回的响应不包含有效内容（content 或 reasoning_content 都为空，且没有 reasoning_content 字段）。"
                        "累积内容: content='%s', reasoning='%s', has_reasoning_field=%s",
                        accumulated_content[:100],
                        accumulated_reasoning[:100],
                        has_reasoning_field
                    )
                    raise AppException(
                        "大模型返回的响应不包含有效内容。",
                        error_code="invalid_llm_response",
                    )
        else:
            # 当复用现有完成结果时，仍然需要处理已执行的工具调用
            completion_result = existing_tool_completion_result

        final_result = await self.persist_completion_result(
            session_id=str(state["session_id"]),
            completion_result=completion_result,  # type: ignore
            executed_tool_calls=executed_tool_calls,
            used_session_memory=prepared_context.used_session_memory,
        )
        return {
            **prepared_context_state,
            "final_result": final_result,
        }

    async def _prepare_context(
        self,
        *,
        execution_request: ChatExecutionRequest,
        answer_instruction: str | None,
        executor_results_context: str | None,
        knowledge_context: str | None,
        route_context: str | None,
        mcp_context: str | None,
        traffic_context: str | None,
        service_context: str | None,
        report_context: str | None,
    ) -> PreparedContext:
        """根据执行请求和节点状态准备上下文。"""

        recent_messages: list[LlmInputMessage] = []
        memory_summary: str | None = None
        if execution_request.need_session_memory and execution_request.session_id is not None:
            memory_snapshot = await self._memory_manager.load_snapshot(execution_request.session_id)
            recent_messages = await self._load_recent_messages(execution_request.session_id)
            memory_summary = memory_snapshot.summary

        return self._context_builder.build_context(
            input_messages=execution_request.input_messages,
            recent_messages=recent_messages,
            memory_summary=memory_summary,
            need_session_memory=execution_request.need_session_memory,
            answer_instruction=answer_instruction,
            executor_results_context=executor_results_context,
            knowledge_context=knowledge_context,
            route_context=route_context,
            mcp_context=mcp_context,
            traffic_context=traffic_context,
            service_context=service_context,
            report_context=report_context,
        )

    async def prepare_context_state(self, state: AgentState) -> dict[str, PreparedContext]:
        """根据状态准备上下文。"""

        execution_request = self._build_execution_request_from_state(state)
        answer_instruction = self._resolve_answer_instruction(state)
        LOGGER.info(
            "Answer prompt selected: category=%s prompt=%s current_step_id=%s",
            state.get("primary_category", "general"),
            self._resolve_answer_prompt_name(state),
            state.get("current_step_id"),
        )
        prepared_context = await self._prepare_context(
            execution_request=execution_request,
            answer_instruction=answer_instruction,
            executor_results_context=self._build_executor_results_context(
                state.get("step_results", {})
                if isinstance(state.get("step_results", {}), dict)
                else {}
            ),
            knowledge_context=state.get("knowledge_context"),
            route_context=state.get("route_context"),
            mcp_context=state.get("mcp_context"),
            traffic_context=state.get("traffic_context"),
            service_context=state.get("service_context"),
            report_context=state.get("report_context"),
        )
        return {"prepared_context": prepared_context}

    async def persist_completion_result(
        self,
        *,
        session_id: str,
        completion_result: AIMessage,
        executed_tool_calls: list[ExecutedToolCall],
        used_session_memory: bool,
    ) -> ChatTurnResult:
        """持久化本轮完整补全结果。"""

        return await self._persist_completion_result(
            session_id=session_id,
            completion_result=completion_result,
            executed_tool_calls=executed_tool_calls,
            used_session_memory=used_session_memory,
        )

    async def persist_stream_result(
        self,
        *,
        session_id: str,
        completion_result: AIMessage,
        used_session_memory: bool,
    ) -> ChatTurnResult:
        """持久化流式路径最终得到的模型输出。"""

        return await self._persist_completion_result(
            session_id=session_id,
            completion_result=completion_result,
            executed_tool_calls=[],
            used_session_memory=used_session_memory,
        )


    async def persist_assistant_tool_calls(
        self,
        *,
        session_id: str,
        completion_result: AIMessage,
    ) -> None:
        """持久化助手的工具调用消息。"""

        if not completion_result.tool_calls:
            return

        await self._message_repository.create(
            message_id=self._generate_identifier(),
            session_id=session_id,
            role="assistant",
            content=self._extract_message_text(completion_result),
            message_metadata={
                "tool_calls": self._serialize_tool_calls(completion_result),
            },
        )

    def build_execution_request_from_state(self, state: AgentState) -> ChatExecutionRequest:
        """从状态构建执行请求。"""

        return self._build_execution_request_from_state(state)

    def _build_execution_request_from_state(self, state: AgentState) -> ChatExecutionRequest:
        """从状态构建执行请求。
        
        优先级：请求中的 model_name > OPENAI_MODEL 配置
        """

        # 优先使用请求中的 model_name，次选 OPENAI_MODEL 配置
        model_name = state.get("model_name") or self._settings.openai_model

        return ChatExecutionRequest(
            session_id=state.get("session_id"),
            need_session_memory=state.get("need_session_memory", False),
            user_id=state.get("user_id"),
            latest_user_message=state.get("latest_user_message", ""),
            input_messages=state.get("input_messages", []),
            model_name=model_name,
            requested_tool_names=state.get("requested_tool_names"),
            tool_choice=state.get("tool_choice"),
            enable_thinking=state.get("enable_thinking"),
        )

    def _generate_identifier(self) -> str:
        """生成唯一标识符。"""

        return uuid4().hex

    @staticmethod
    def _build_executor_results_context(step_results: dict[str, ExecutorResult]) -> str | None:
        """构建执行器结果上下文。"""

        if not step_results:
            return None

        parts = ["以下是当前执行节点返回的结构化结果，请基于这些结果组织最终回答："]
        for step_id, result in step_results.items():
            result_summary = "\n".join(
                f"- {k}: {v}"
                for k, v in (result.normalized_result or {}).items()
                if k != "raw_result"
            )
            parts.append(
                f"[{step_id}] executor={result.executor} success={result.is_success} "
                f"sources={result.sources}\n{result_summary}\n{result.summary}"
            )

        return "\n\n".join(parts)

    def _extract_executed_tool_calls(self, state: AgentState) -> list[ExecutedToolCall]:
        """从状态中提取已执行的工具调用。"""

        raw_executed_tool_calls = state.get("executed_tool_calls")
        if not raw_executed_tool_calls:
            return []

        if isinstance(raw_executed_tool_calls, list):
            return [
                tool_call
                for tool_call in raw_executed_tool_calls
                if isinstance(tool_call, ExecutedToolCall)
            ]

        return []

    def should_generate_summary(self, state: AgentState) -> bool:
        """判断是否需要生成摘要。"""

        return self._should_generate_summary(state)

    def _should_generate_summary(self, state: AgentState) -> bool:
        """判断是否需要生成摘要。"""

        current_step_id = state.get("current_step_id")
        if current_step_id is None:
            return False

        step_results = state.get("step_results", {})
        if not isinstance(step_results, dict):
            return False

        current_step = get_execution_step(state, step_id=str(current_step_id))
        if current_step is None:
            return False

        # 如果当前步骤是多步骤计划的一部分，需要重新生成摘要
        return len(step_results) > 1

    async def _load_recent_messages(self, session_id: str) -> list[LlmInputMessage]:
        """加载最近的消息。"""

        recent_entities = await self._message_repository.list_by_session(
            session_id,
            limit=RECENT_CONTEXT_WINDOW_SIZE,
        )
        # 使用 message_entity_to_input_message 函数将实体转换为 LlmInputMessage
        from app.agent.context_builder import message_entity_to_input_message

        return [message_entity_to_input_message(entity) for entity in recent_entities]

    async def _persist_completion_result(
        self,
        *,
        session_id: str,
        completion_result: AIMessage,
        executed_tool_calls: list[ExecutedToolCall],
        used_session_memory: bool,
    ) -> ChatTurnResult:
        """持久化补全结果。"""

        await self._message_repository.create(
            message_id=self._generate_identifier(),
            session_id=session_id,
            role="assistant",
            content=self._extract_message_text(completion_result),
            message_metadata={
                "tool_calls": self._serialize_tool_calls(completion_result),
                "response_metadata": completion_result.response_metadata,
                "usage_metadata": completion_result.usage_metadata,
            },
        )

        usage_metadata = completion_result.usage_metadata or {}
        response_metadata = completion_result.response_metadata or {}
        prompt_tokens = int(usage_metadata.get("input_tokens") or 0)
        completion_tokens = int(usage_metadata.get("output_tokens") or 0)
        total_tokens = int(
            usage_metadata.get("total_tokens") or (prompt_tokens + completion_tokens)
        )
        return ChatTurnResult(
            session_id=session_id,
            content=self._extract_message_text(completion_result),
            model_name=str(
                response_metadata.get("model_name") or response_metadata.get("model") or ""
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(
                response_metadata.get("finish_reason")
                or ("tool_calls" if completion_result.tool_calls else "stop")
            ),
            route="answer",
            reasoning_content=self._extract_reasoning_text(completion_result),
            tool_calls=list(executed_tool_calls),
            used_session_memory=used_session_memory,
        )

    @staticmethod
    def _extract_message_text(message: AIMessage) -> str:
        """将 AIMessage 的内容稳定归一化为纯文本。"""

        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            text_parts: list[str] = []
            for part in message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return str(message.content)

    @staticmethod
    def _extract_reasoning_text(message: AIMessage) -> str | None:
        """Extract provider-specific reasoning text from additional kwargs when present."""

        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
        reasoning_content = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content or None
        return None

    @staticmethod
    def _serialize_tool_calls(message: AIMessage) -> list[dict[str, object]]:
        """将 LangChain tool_calls 归一化为内部消息元数据格式。"""

        return [
            {
                "tool_call_id": str(tool_call.get("id", "")),
                "tool_name": str(tool_call.get("name", "")),
                "arguments": (
                    dict(tool_call.get("args", {}))
                    if isinstance(tool_call.get("args"), dict)
                    else {}
                ),
            }
            for tool_call in (message.tool_calls or [])
        ]

    @staticmethod
    def _resolve_answer_instruction(state: AgentState) -> str:
        """根据主分类选择最终回答阶段的提示词。"""

        category = AnswerNode._resolve_answer_topic(state)
        if category == "policy":
            return POLICY_SUMMARY_PROMPT
        if category == "route_planning":
            return ROUTE_SUMMARY_PROMPT
        if category == "traffic_status":
            return TRAFFIC_SUMMARY_PROMPT
        if category == "service_area":
            return SERVICE_SUMMARY_PROMPT
        if category == "network_report":
            return NETWORK_REPORT_SUMMARY_PROMPT
        return GENERAL_ANSWER_PROMPT

    @staticmethod
    def _resolve_answer_prompt_name(state: AgentState) -> str:
        category = AnswerNode._resolve_answer_topic(state)
        return _PROMPT_NAME_BY_CATEGORY.get(category, "GENERAL_ANSWER_PROMPT")

    @staticmethod
    def _resolve_answer_topic(state: AgentState) -> str:
        """结合原始问题和已完成执行结果，选择最合适的回答模板。"""

        latest_user_message = str(state.get("latest_user_message", ""))
        normalized_message = latest_user_message.strip()

        step_results = state.get("step_results", {})
        executed_executors: set[str] = set()
        if isinstance(step_results, dict):
            for result in step_results.values():
                if isinstance(result, ExecutorResult):
                    executed_executors.add(result.executor)

        if "report" in executed_executors or AnswerNode._looks_like_report_query(normalized_message):
            return "network_report"
        if "traffic" in executed_executors or AnswerNode._looks_like_traffic_query(normalized_message):
            return "traffic_status"
        if "service" in executed_executors or AnswerNode._looks_like_service_query(normalized_message):
            return "service_area"
        if "rag" in executed_executors or AnswerNode._looks_like_policy_query(normalized_message):
            return "policy"
        if "route" in executed_executors or AnswerNode._looks_like_route_query(normalized_message):
            return "route_planning"

        category = str(state.get("primary_category", "general"))
        if category in _PROMPT_NAME_BY_CATEGORY:
            return category
        return "general"

    @staticmethod
    def _looks_like_traffic_query(message: str) -> bool:
        return any(keyword in message for keyword in _TRAFFIC_KEYWORDS)

    @staticmethod
    def _looks_like_service_query(message: str) -> bool:
        return any(keyword in message for keyword in _SERVICE_KEYWORDS)

    @staticmethod
    def _looks_like_policy_query(message: str) -> bool:
        return any(keyword in message for keyword in _POLICY_KEYWORDS)

    @staticmethod
    def _looks_like_route_query(message: str) -> bool:
        return any(keyword in message for keyword in _ROUTE_KEYWORDS)

    @staticmethod
    def _looks_like_report_query(message: str) -> bool:
        return any(keyword in message for keyword in _REPORT_KEYWORDS)
