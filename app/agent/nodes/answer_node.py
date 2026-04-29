"""回答节点模块。

负责构建模型上下文、执行普通回答补全，并持久化本轮输出。
当前阶段不负责工具循环编排，这部分能力由独立的 tool_node 承担。
"""

from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.answer_prompts import (
    COMPOSITE_ANSWER_PROMPT,
    GENERAL_ANSWER_PROMPT,
    NETWORK_REPORT_SUMMARY_PROMPT,
    POLICY_SUMMARY_PROMPT,
    ROUTE_SUMMARY_PROMPT,
    SERVICE_SUMMARY_PROMPT,
    TRAFFIC_SUMMARY_PROMPT,
)
from app.agent.brief_answer_prompts import (
    BRIEF_COMPOSITE_ANSWER_PROMPT,
    BRIEF_GENERAL_ANSWER_PROMPT,
    BRIEF_POLICY_SUMMARY_PROMPT,
    BRIEF_ROUTE_SUMMARY_PROMPT,
    BRIEF_SERVICE_SUMMARY_PROMPT,
    BRIEF_TRAFFIC_SUMMARY_PROMPT,
)
from app.agent.context_builder import (
    MAX_REPORT_CONTEXT_TOKENS,
    MAX_SCHEDULED_REPORT_CONTEXT_TOKENS,
    ContextBuilder,
    estimate_messages_tokens,
    truncate_text_to_token_budget,
)
from app.agent.history_utils import MAX_CONTEXT_MESSAGES
from app.agent.network_report_renderer import (
    RenderedNetworkReport,
    build_network_report_render_result,
    coerce_executor_result,
)
from app.agent.state import (
    AgentState,
    ChatExecutionRequest,
    ChatTurnResult,
    ExecutorResult,
    PreparedContext,
    get_execution_step,
)
from app.agent.tool_traffic_intent import looks_like_traffic_query_v1
from app.clients.llm_client import LlmClient, LlmInputMessage
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logger import get_logger
from app.memory.manager import MemoryManager
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall, ToolRegistry

LOGGER = get_logger(__name__)

RECENT_CONTEXT_WINDOW_SIZE = MAX_CONTEXT_MESSAGES
NETWORK_REPORT_BROADCAST_PROMPT = """你是高速路网播报总结助手。

你会收到完整的 report_content 上下文，它已经包含查询时间、拥堵汇总、主线管制和收费站管制的结构化文本。

你的任务只有一个：基于整个上下文,对整体情况做出总结。

输出要求：
- 只输出播报正文，不要输出 Markdown 表格，不要重复字段名。
- 不要输出“AI播报总结：”“总结如下”等前缀。
- 优先概括全网整体态势，再点出最需要关注的异常类型或重点路段。
- 不要编造上下文里没有的信息，不要输出内部字段名或分析过程。
- 字数控制在 100-200 字。
- 禁止出现具体时间信息。
- 禁止输出道路事故明细。
- 优先概括整体交通管制、收费站管控、拥堵/缓行情况。

"""
_PROMPT_NAME_BY_CATEGORY = {
    "composite": "COMPOSITE_ANSWER_PROMPT",
    "policy": "POLICY_SUMMARY_PROMPT",
    "route_planning": "ROUTE_SUMMARY_PROMPT",
    "traffic_status": "TRAFFIC_SUMMARY_PROMPT",
    "service_area": "SERVICE_SUMMARY_PROMPT",
    "network_report": "NETWORK_REPORT_SUMMARY_PROMPT",
    "general": "GENERAL_ANSWER_PROMPT",
}
_BRIEF_PROMPT_NAME_BY_CATEGORY = {
    "composite": "BRIEF_COMPOSITE_ANSWER_PROMPT",
    "policy": "BRIEF_POLICY_SUMMARY_PROMPT",
    "route_planning": "BRIEF_ROUTE_SUMMARY_PROMPT",
    "traffic_status": "BRIEF_TRAFFIC_SUMMARY_PROMPT",
    "service_area": "BRIEF_SERVICE_SUMMARY_PROMPT",
    "general": "BRIEF_GENERAL_ANSWER_PROMPT",
}

# _TRAFFIC_KEYWORDS = (
#     "堵不堵",
#     "拥堵",
#     "路况",
#     "堵吗",
#     "缓行",
#     "施工",
#     "事故",
#     "封闭",
#     "管制",
#     "通行",
#     "是否畅通",
#     "是否拥堵",
#     "会不会堵",
# )
_SERVICE_KEYWORDS = (
    "服务区",
    "充电桩",
    "充电",
    "加油",
    "休息区",
    "配套",
    "快充"
    "快充"
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
_TOLL_KEYWORDS = (
    "过路费",
    "通行费",
    "收费",
    "免收费",
    "高速费",
    "费率",
    "免费时段",
    "节假日",
    "出口时间",
    "入口时间",
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

        network_report_render = self._build_network_report_render_result(state)
        if AnswerNode._resolve_answer_topic(state) == "network_report":
            step_results = state.get("step_results")
            if isinstance(step_results, dict):
                step_result_keys = ",".join(sorted(str(step_id) for step_id in step_results))
            else:
                step_result_keys = type(step_results).__name__
            LOGGER.info(
                (
                    "Network report probe: requested_scheduled_route=%s scheduled_route=%s "
                    "route=%s current_step_id=%s step_result_keys=%s "
                    "prepared_context_present=%s render_ready=%s report_context_budget=%s"
                ),
                state.get("requested_scheduled_route"),
                state.get("scheduled_route"),
                state.get("route"),
                state.get("current_step_id"),
                step_result_keys,
                isinstance(state.get("prepared_context"), PreparedContext),
                network_report_render is not None,
                self._resolve_report_context_token_budget(state),
            )
            if network_report_render is None:
                LOGGER.warning(
                    "Network report renderer returned None; falling back to generic answer path."
                )
        if network_report_render is not None:
            prepared_context_state = self._prepare_network_report_context_state(state)
            prepared_context = prepared_context_state["prepared_context"]
            completion_result = state.get("tool_completion_result")
            executed_tool_calls = self._extract_executed_tool_calls(state)
            include_table = self._should_include_network_report_table(state)
            completion_result = await self._build_network_report_completion_result(
                state=state,
                prepared_context=prepared_context,
                render_result=network_report_render,
                include_table=include_table,
            )
        else:
            prepared_context = state.get("prepared_context")
            if isinstance(prepared_context, PreparedContext):
                prepared_context_state = {"prepared_context": prepared_context}
            else:
                prepared_context_state = await self.prepare_context_state(state)
                prepared_context = prepared_context_state["prepared_context"]

            completion_result = self._build_empty_executor_fallback_message(state)
            if completion_result is None:
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
                    api_key=self._settings.resolved_openai_api_key_value,
                    base_url=self._settings.resolved_openai_base_url,
                    timeout_seconds=self._settings.openai_timeout_seconds,
                    enable_thinking=execution_request.enable_thinking,
                    is_stream=True,
                )

                completion_result = None
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
                    if hasattr(chunk, "content") and chunk.content:
                        accumulated_content += chunk.content
                    if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                        if "reasoning_content" in chunk.additional_kwargs:
                            has_reasoning_field = True
                            reasoning = chunk.additional_kwargs["reasoning_content"]
                            if reasoning:  # 只有非空时才累积
                                accumulated_reasoning += reasoning

                # 检查是否有实际内容或reasoning字段
                has_content_or_reasoning = bool(
                    accumulated_content or accumulated_reasoning or has_reasoning_field
                )

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
                    if (
                        hasattr(completion_result, "additional_kwargs")
                        and completion_result.additional_kwargs
                        and "reasoning_content" in completion_result.additional_kwargs
                        and completion_result.additional_kwargs["reasoning_content"]
                    ):
                        has_content_or_reasoning = True

                    if not has_content_or_reasoning:
                        LOGGER.warning(
                            "大模型返回的响应不包含有效内容（content 或 reasoning_content 都为空，且没有 reasoning_content 字段）。"
                            "累积内容: content='%s', reasoning='%s', has_reasoning_field=%s",
                            accumulated_content[:100],
                            accumulated_reasoning[:100],
                            has_reasoning_field,
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

    @staticmethod
    def _resolve_report_context_token_budget(state: AgentState) -> int:
        """根据是否显式调度 report 决定 report_content 预算。"""

        return (
            MAX_SCHEDULED_REPORT_CONTEXT_TOKENS
            if (
                state.get("requested_scheduled_route") == "report"
                or state.get("forced_route") == "report"
                or state.get("scheduled_route") == "report"
            )
            else MAX_REPORT_CONTEXT_TOKENS
        )

    @staticmethod
    def _prepare_network_report_context_state(state: AgentState) -> dict[str, PreparedContext]:
        prepared_context = state.get("prepared_context")
        if isinstance(prepared_context, PreparedContext):
            return {"prepared_context": prepared_context}

        report_context = state.get("report_context")
        report_context_max_tokens = AnswerNode._resolve_report_context_token_budget(state)
        truncated_report_context = (
            truncate_text_to_token_budget(
                report_context,
                max_tokens=report_context_max_tokens,
                label="report_content",
            )
            if isinstance(report_context, str)
            else None
        )
        return {
            "prepared_context": PreparedContext(
                messages=[],
                used_session_memory=False,
                report_context=truncated_report_context,
            )
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
        report_context_max_tokens: int,
        max_turns: int,
    ) -> PreparedContext:
        """根据执行请求和节点状态准备上下文。"""

        recent_messages: list[LlmInputMessage] = []
        if execution_request.need_session_memory and execution_request.session_id is not None:
            recent_messages = await self._load_recent_messages(execution_request.session_id)

        return self._context_builder.build_context(
            input_messages=execution_request.input_messages,
            recent_messages=recent_messages,
            memory_summary=None,
            need_session_memory=execution_request.need_session_memory,
            max_turns=max_turns,
            model_name=execution_request.model_name,
            answer_instruction=answer_instruction,
            executor_results_context=executor_results_context,
            knowledge_context=knowledge_context,
            route_context=route_context,
            mcp_context=mcp_context,
            traffic_context=traffic_context,
            service_context=service_context,
            report_context=report_context,
            report_context_max_tokens=report_context_max_tokens,
        )

    async def prepare_context_state(self, state: AgentState) -> dict[str, PreparedContext]:
        """根据状态准备上下文。"""

        execution_request = self._build_execution_request_from_state(state)
        answer_instruction = self._resolve_answer_instruction(state)
        max_turns = 1
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
            report_context_max_tokens=self._resolve_report_context_token_budget(state),
            max_turns=max_turns,
        )
        LOGGER.info(
            "Prepared context estimate: model=%s estimated_prompt_tokens=%s message_count=%s used_session_memory=%s",
            execution_request.model_name or self._settings.openai_model,
            prepared_context.estimated_prompt_tokens,
            len(prepared_context.messages),
            prepared_context.used_session_memory,
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
        model_name = state.get("model_name") or self._settings.resolved_openai_model

        return ChatExecutionRequest(
            session_id=state.get("session_id"),
            need_session_memory=state.get("need_session_memory", False),
            user_id=state.get("user_id"),
            latest_user_message=state.get("latest_user_message", ""),
            input_messages=state.get("input_messages", []),
            model_name=model_name,
            requested_tool_names=state.get("requested_tool_names"),
            tool_choice=state.get("tool_choice"),
            scheduled_route=state.get("scheduled_route"),
            enable_thinking=state.get("enable_thinking"),
            brief_answer=bool(state.get("brief_answer", True)),
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
            compact_result = AnswerNode._compact_executor_result(result)
            result_summary = "\n".join(f"- {k}: {v}" for k, v in compact_result.items())
            details_block = result_summary if result_summary else "- no_compact_fields: true"
            parts.append(
                f"[{step_id}] executor={result.executor} success={result.is_success} "
                f"sources={result.sources}\n{details_block}\n{result.summary}"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _compact_executor_result(result: ExecutorResult) -> dict[str, object]:
        """压缩执行结果，避免把已在专用上下文中出现的大块明细重复注入。"""

        normalized_result = result.normalized_result or {}
        if not isinstance(normalized_result, dict):
            return {}

        keep_keys_by_executor: dict[str, set[str]] = {
            "route": {"routes_count"},
            "traffic": {"matched_road_count", "event_count"},
            "service": {"result_count"},
            "rag": {"result_count"},
            "report": set(),
            "mcp": set(),
            "tool": set(),
            "answer": set(),
        }
        allowed_keys = keep_keys_by_executor.get(result.executor, set())

        compact_result: dict[str, object] = {}
        for key, value in normalized_result.items():
            if key not in allowed_keys:
                continue
            compact_result[key] = value
        return compact_result

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
    def _build_network_report_render_result(state: AgentState) -> RenderedNetworkReport | None:
        answer_topic = AnswerNode._resolve_answer_topic(state)
        if answer_topic != "network_report":
            LOGGER.info(
                "Network report renderer skipped: answer_topic=%s scheduled_route=%s current_step_id=%s",
                answer_topic,
                state.get("scheduled_route"),
                state.get("current_step_id"),
            )
            return None

        step_results = state.get("step_results", {})
        if not isinstance(step_results, dict):
            LOGGER.warning(
                "Network report renderer skipped: step_results is not a dict, actual_type=%s",
                type(step_results).__name__,
            )
            return None
        typed_step_results = {}
        for step_id, result in step_results.items():
            if not isinstance(step_id, str):
                continue
            normalized_result = coerce_executor_result(step_id, result)
            if normalized_result is not None:
                typed_step_results[step_id] = normalized_result
        if not typed_step_results:
            LOGGER.warning(
                "Network report renderer skipped: no usable executor results, step_result_keys=%s",
                ",".join(sorted(str(step_id) for step_id in step_results)),
            )
            return None
        render_result = build_network_report_render_result(typed_step_results)
        if render_result is None:
            LOGGER.warning(
                "Network report renderer returned None after coercion, step_result_keys=%s",
                ",".join(sorted(str(step_id) for step_id in typed_step_results)),
            )
            return None
        LOGGER.info(
            "Network report renderer ready: rows=%s summary_chars=%s step_result_keys=%s",
            len(render_result.rows),
            len(render_result.summary),
            ",".join(sorted(str(step_id) for step_id in typed_step_results)),
        )
        return render_result

    async def _build_network_report_completion_result(
        self,
        *,
        state: AgentState,
        prepared_context: PreparedContext,
        render_result: RenderedNetworkReport,
        include_table: bool,
    ) -> AIMessage:
        summary_message = await self._generate_network_report_broadcast_summary(
            state=state,
            prepared_context=prepared_context,
        )
        if summary_message is None:
            if include_table:
                return AIMessage(
                    content=render_result.to_markdown(),
                    response_metadata={
                        "finish_reason": "stop",
                        "model_name": "network-report-renderer",
                    },
                    usage_metadata={
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                )
            return AIMessage(
                content=render_result.summary,
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": "network-report-renderer",
                },
                usage_metadata={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )

        summary_text = self._normalize_network_report_summary_text(
            self._extract_message_text(summary_message)
        ) or render_result.summary
        summary_text = AnswerNode._strip_network_report_summary_prefix(summary_text)
        if not include_table:
            return AIMessage(
                content=summary_text,
                response_metadata={
                    **(summary_message.response_metadata or {}),
                    "finish_reason": str(
                        (summary_message.response_metadata or {}).get("finish_reason") or "stop"
                    ),
                },
                usage_metadata=summary_message.usage_metadata,
                additional_kwargs=getattr(summary_message, "additional_kwargs", None) or {},
            )
        return AIMessage(
            content=f"{summary_text}\n\n{render_result.table_markdown}",
            response_metadata={
                **(summary_message.response_metadata or {}),
                "finish_reason": str(
                    (summary_message.response_metadata or {}).get("finish_reason") or "stop"
                ),
            },
            usage_metadata=summary_message.usage_metadata,
            additional_kwargs=getattr(summary_message, "additional_kwargs", None) or {},
        )

    async def _generate_network_report_broadcast_summary(
        self,
        *,
        state: AgentState,
        prepared_context: PreparedContext,
    ) -> AIMessage | None:
        report_context = prepared_context.report_context or state.get("report_context")
        if not isinstance(report_context, str) or not report_context.strip():
            return None
        report_context_max_tokens = self._resolve_report_context_token_budget(state)
        report_context = (
            truncate_text_to_token_budget(
                report_context,
                max_tokens=report_context_max_tokens,
                label="report_content",
            )
            or ""
        )

        execution_request = self.build_execution_request_from_state(state)
        summary_request = "请基于完整报表上下文输出1-2句路网播报总结。"
        summary_messages = [
            LlmInputMessage(role="system", content=NETWORK_REPORT_BROADCAST_PROMPT),
            LlmInputMessage(role="system", content=report_context),
            LlmInputMessage(role="user", content=summary_request),
        ]
        estimated_prompt_tokens = estimate_messages_tokens(
            summary_messages,
            model_name=execution_request.model_name,
        )
        LOGGER.info(
            "Network report summary prompt estimate: model=%s estimated_prompt_tokens=%s message_count=%s",
            execution_request.model_name or self._settings.openai_model,
            estimated_prompt_tokens,
            len(summary_messages),
        )
        return await self._llm_client.create_chat_completion(
            messages=summary_messages,
            model_name=execution_request.model_name,
            api_key=self._settings.resolved_openai_api_key_value,
            base_url=self._settings.resolved_openai_base_url,
            timeout_seconds=self._settings.openai_timeout_seconds,
            enable_thinking=execution_request.enable_thinking,
        )

    @staticmethod
    def _should_include_network_report_table(state: AgentState) -> bool:
        del state
        return True

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
    def _normalize_network_report_summary_text(text: str) -> str:
        normalized_text = text.strip()
        for prefix in (
            "AI播报总结：",
            "AI播报总结:",
            "播报总结：",
            "播报总结:",
            "总结：",
            "总结:",
        ):
            if normalized_text.startswith(prefix):
                normalized_text = normalized_text[len(prefix) :].strip()

        normalized_lines: list[str] = []
        for line in normalized_text.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                if normalized_lines:
                    break
                continue
            if stripped_line.startswith("|") or stripped_line.startswith("```"):
                break
            normalized_lines.append(stripped_line)
        return " ".join(normalized_lines).strip()

    @staticmethod
    def _strip_network_report_summary_prefix(text: str) -> str:
        normalized_text = text.strip()
        for prefix in (
            "AI播报总结：",
            "AI播报总结:",
            "播报总结：",
            "播报总结:",
            "总结：",
            "总结:",
        ):
            if normalized_text.startswith(prefix):
                normalized_text = normalized_text[len(prefix) :].strip()
        return normalized_text

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

        brief_answer = AnswerNode._should_use_brief_answer_prompt(state)
        if AnswerNode._should_use_composite_prompt(state):
            return AnswerNode._build_composite_answer_instruction(
                state,
                brief_answer=brief_answer,
            )

        category = AnswerNode._resolve_answer_topic(state)
        if category == "network_report":
            return NETWORK_REPORT_SUMMARY_PROMPT
        if brief_answer:
            if category == "policy":
                return BRIEF_POLICY_SUMMARY_PROMPT
            if category == "route_planning" or AnswerNode._should_use_route_summary_prompt(state):
                return AnswerNode._build_route_answer_instruction(state, brief_answer=True)
            if category == "traffic_status":
                return BRIEF_TRAFFIC_SUMMARY_PROMPT
            if category == "service_area":
                return BRIEF_SERVICE_SUMMARY_PROMPT
            return BRIEF_GENERAL_ANSWER_PROMPT
        if category == "policy":
            return POLICY_SUMMARY_PROMPT
        if category == "route_planning" or AnswerNode._should_use_route_summary_prompt(state):
            return AnswerNode._build_route_answer_instruction(state)
        if category == "traffic_status":
            return TRAFFIC_SUMMARY_PROMPT
        if category == "service_area":
            return SERVICE_SUMMARY_PROMPT
        return GENERAL_ANSWER_PROMPT

    @staticmethod
    def _resolve_answer_prompt_name(state: AgentState) -> str:
        brief_answer = AnswerNode._should_use_brief_answer_prompt(state)
        if AnswerNode._should_use_composite_prompt(state):
            return (
                "BRIEF_COMPOSITE_ANSWER_PROMPT"
                if brief_answer
                else "COMPOSITE_ANSWER_PROMPT"
            )

        category = AnswerNode._resolve_answer_topic(state)
        if category == "network_report":
            return "NETWORK_REPORT_SUMMARY_PROMPT"
        if brief_answer:
            if category == "route_planning" or AnswerNode._should_use_route_summary_prompt(state):
                return "BRIEF_ROUTE_SUMMARY_PROMPT"
            return _BRIEF_PROMPT_NAME_BY_CATEGORY.get(category, "BRIEF_GENERAL_ANSWER_PROMPT")
        if category == "route_planning" or AnswerNode._should_use_route_summary_prompt(state):
            return "ROUTE_SUMMARY_PROMPT"
        return _PROMPT_NAME_BY_CATEGORY.get(category, "GENERAL_ANSWER_PROMPT")

    @staticmethod
    def _should_use_brief_answer_prompt(state: AgentState) -> bool:
        return bool(state.get("brief_answer", True))

    @staticmethod
    def _resolve_answer_topic(state: AgentState) -> str:
        """结合原始问题和已完成执行结果，选择最合适的回答模板。"""

        latest_user_message = str(state.get("latest_user_message", ""))
        normalized_message = latest_user_message.strip()

        step_results = state.get("step_results", {})
        executed_executors: set[str] = set()
        if isinstance(step_results, dict):
            for step_id, result in step_results.items():
                if isinstance(step_id, str):
                    normalized_result = coerce_executor_result(step_id, result)
                    if normalized_result is not None:
                        executed_executors.add(normalized_result.executor)

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
    def _should_use_composite_prompt(state: AgentState) -> bool:
        """复合查询优先走综合模板，而不是回落到单一分类模板。"""

        executed_executors = AnswerNode._collect_executed_executors(state)
        if len(executed_executors) >= 2:
            return True

        execution_plan = state.get("execution_plan")
        if isinstance(execution_plan, object) and hasattr(execution_plan, "steps"):
            planned_executors = {
                step.executor
                for step in execution_plan.steps
                if step.executor != "answer"
            }
            if len(planned_executors) >= 2:
                return True

        return False

    @staticmethod
    def _collect_executed_executors(state: AgentState) -> set[str]:
        step_results = state.get("step_results", {})
        executed_executors: set[str] = set()
        if isinstance(step_results, dict):
            for step_id, result in step_results.items():
                if not isinstance(step_id, str):
                    continue
                normalized_result = coerce_executor_result(step_id, result)
                if normalized_result is not None and normalized_result.executor != "answer":
                    executed_executors.add(normalized_result.executor)
        return executed_executors

    @staticmethod
    def _build_composite_answer_instruction(
        state: AgentState,
        *,
        brief_answer: bool = False,
    ) -> str:
        focus = AnswerNode._resolve_planner_focus(state) or AnswerNode._resolve_answer_focus(state)
        executed_executors = sorted(AnswerNode._collect_executed_executors(state))

        instruction_parts = [
            BRIEF_COMPOSITE_ANSWER_PROMPT if brief_answer else COMPOSITE_ANSWER_PROMPT
        ]
        if focus:
            instruction_parts.append(f"本轮回答焦点：{focus}")
        if executed_executors:
            instruction_parts.append(
                "本轮已完成能力："
                + "、".join(executed_executors)
                + "。请优先综合这些结果，不要只围绕其中一个模块作答。"
            )
        return "\n\n".join(instruction_parts)

    @staticmethod
    def _build_route_answer_instruction(
        state: AgentState,
        *,
        brief_answer: bool = False,
    ) -> str:
        """优先根据用户问题正则判断焦点，metadata.focus 作为兜底。"""

        focus = (
            AnswerNode._resolve_route_summary_focus(state)
            or AnswerNode._resolve_planner_focus(state)
        )
        prompt = BRIEF_ROUTE_SUMMARY_PROMPT if brief_answer else ROUTE_SUMMARY_PROMPT
        return prompt.format(focus=focus or "未提供")

    @staticmethod
    def _looks_like_service_query(message: str) -> bool:
        return any(keyword in message for keyword in _SERVICE_KEYWORDS)

    @staticmethod
    def _looks_like_policy_query(message: str) -> bool:
        return any(keyword in message for keyword in _POLICY_KEYWORDS)


    @staticmethod
    def _looks_like_report_query(message: str) -> bool:
        return any(keyword in message for keyword in _REPORT_KEYWORDS)

    @staticmethod
    def _resolve_planner_focus(state: AgentState) -> str | None:
        """读取 planner 为当前 answer 步骤写入的 metadata.focus。"""

        current_step_id = state.get("current_step_id")
        if isinstance(current_step_id, str):
            current_step = get_execution_step(state, step_id=current_step_id)
            focus = AnswerNode._extract_step_focus(current_step)
            if focus:
                return focus

        answer_step = get_execution_step(state, executor="answer")
        return AnswerNode._extract_step_focus(answer_step)

    @staticmethod
    def _resolve_route_summary_focus(state: AgentState) -> str | None:
        """优先根据用户当前问题和已执行上下文推断 route summary 的焦点。"""

        message = str(state.get("latest_user_message", "")).strip()
        executed_executors = AnswerNode._collect_executed_executors(state)
        execution_plan = state.get("execution_plan")
        planned_executors: set[str] = set()
        if isinstance(execution_plan, object) and hasattr(execution_plan, "steps"):
            planned_executors = {
                step.executor
                for step in execution_plan.steps
                if step.executor != "answer"
            }
        has_route_context = "route" in executed_executors or "route" in planned_executors

        if any(keyword in message for keyword in _TOLL_KEYWORDS):
            return "收费判断。优先回答是否收费、免费时间窗口、按什么时间规则判定，以及还缺哪些条件。"
        if has_route_context:
            if AnswerNode._looks_like_route_query(message):
                return "路线推荐与关键路况。优先按“推荐路线 -> 预计时长 -> 关键路况 -> 口述式总结”组织回答。"
            return "路况与管制。优先给整体通行判断，再补充关键路段、收费站、匝道和必要的路线补充。"
        if AnswerNode._looks_like_traffic_query(message):
            return "路况与管制。"
        if AnswerNode._looks_like_service_query(message):
            return "服务区设施。优先回答是否有充电桩、主要配套和繁忙程度。"
        if AnswerNode._looks_like_policy_query(message):
            return "政策规则。优先回答适用范围、判断依据、关键条件和限制。"
        if AnswerNode._looks_like_report_query(message):
            return "路网汇总。优先概括整体态势、变化重点和需要关注的路段。"
        if AnswerNode._looks_like_route_query(message):
            return "出行方案。优先回答推荐路线、预计时长和关键提醒。"
        return None

    @staticmethod
    def _should_use_route_summary_prompt(state: AgentState) -> bool:
        """只要本轮已经有 route 步骤，就优先使用路线回答模板。"""

        step_results = state.get("step_results", {})
        if not isinstance(step_results, dict):
            return False

        for result in step_results.values():
            if isinstance(result, ExecutorResult) and result.executor == "route":
                return True
        return False

    @staticmethod
    def _extract_step_focus(step) -> str | None:
        if step is None:
            return None

        metadata = getattr(step, "metadata", None)
        if not isinstance(metadata, dict):
            return None

        focus = metadata.get("focus")
        if isinstance(focus, str):
            stripped_focus = focus.strip()
            if stripped_focus:
                return stripped_focus
        return None

    @staticmethod
    def _has_route_and_traffic_context(state: AgentState) -> bool:
        executors = AnswerNode._collect_executed_executors(state)
        if {"route", "traffic"}.issubset(executors):
            return True

        execution_plan = state.get("execution_plan")
        if isinstance(execution_plan, object) and hasattr(execution_plan, "steps"):
            planned_executors = {
                step.executor
                for step in execution_plan.steps
                if step.executor != "answer"
            }
            return {"route", "traffic"}.issubset(planned_executors)
        return False

    @staticmethod
    def _resolve_answer_focus(state: AgentState) -> str:
        message = str(state.get("latest_user_message", "")).strip()
        executed_executors = AnswerNode._collect_executed_executors(state)
        execution_plan = state.get("execution_plan")
        planned_executors: set[str] = set()
        if isinstance(execution_plan, object) and hasattr(execution_plan, "steps"):
            planned_executors = {
                step.executor
                for step in execution_plan.steps
                if step.executor != "answer"
            }
        has_route_context = "route" in executed_executors or "route" in planned_executors

        if any(keyword in message for keyword in _TOLL_KEYWORDS):
            return "收费判断。优先回答是否收费、免费时间窗口、按什么时间规则判定，以及还缺哪些条件。"
        if has_route_context:
            if AnswerNode._looks_like_route_query(message):
                return "路线推荐与关键路况。优先按“推荐路线 -> 预计时长 -> 关键路况 -> 口述式总结”组织回答。"
            return "路况与管制。优先给整体通行判断，再补充关键路段、收费站、匝道和必要的路线补充。"
        if AnswerNode._looks_like_route_query(message):
            return "出行方案。优先回答推荐路线、预计时长和关键提醒。"
        if AnswerNode._looks_like_traffic_query(message):
            return "路况与管制。"
        if AnswerNode._looks_like_service_query(message):
            return "服务区设施。优先回答是否有充电桩、主要配套和繁忙程度。"
        if AnswerNode._looks_like_policy_query(message):
            return "政策规则。优先回答适用范围、判断依据、关键条件和限制。"
        if AnswerNode._looks_like_report_query(message):
            return "路网汇总。优先概括整体态势、变化重点和需要关注的路段。"
        if AnswerNode._looks_like_route_query(message):
            return "出行方案。优先回答推荐路线、预计时长和关键提醒。"
        return "综合结论。优先直接回答用户问题，再补充最关键的支撑信息。"

    @staticmethod
    def _looks_like_traffic_query(message: str) -> bool:
        # traffic_keywords = _TRAFFIC_KEYWORDS + (
        #     "堵车",
        #     "怎么样",
        #     "咋样",
        #     "正常通行",
        #     "正常吗",
        #     "看一下",
        #     "可以上吗",
        #     "可以走吗",
        #     "可以走",
        #     "可以上",
        #     "情况"
        #     "看看",
        # )
        # return any(keyword in message for keyword in traffic_keywords)
        return looks_like_traffic_query_v1(message)

    @staticmethod
    def _looks_like_route_query(message: str) -> bool:
        route_keywords = _ROUTE_KEYWORDS + (
            "最快",
            "推荐",
            "走哪条",
            "哪条高速",
            "怎么开",
        )
        return any(keyword in message for keyword in route_keywords)

    @staticmethod
    def _build_empty_executor_fallback_message(state: AgentState) -> AIMessage | None:
        traffic_fallback = AnswerNode._build_empty_traffic_fallback_message(state)
        if traffic_fallback is not None:
            return traffic_fallback
        return None

    @staticmethod
    def _build_empty_traffic_fallback_message(state: AgentState) -> AIMessage | None:
        if AnswerNode._resolve_answer_topic(state) != "traffic_status":
            return None

        step_results = state.get("step_results", {})
        if not isinstance(step_results, dict):
            return None

        traffic_result: ExecutorResult | None = None
        for result in step_results.values():
            if isinstance(result, ExecutorResult) and result.executor == "traffic":
                traffic_result = result

        if traffic_result is None:
            return None

        normalized_result = (
            traffic_result.normalized_result
            if isinstance(traffic_result.normalized_result, dict)
            else {}
        )
        if int(normalized_result.get("result_count") or 0) > 0:
            return None
        if int(normalized_result.get("matched_road_count") or 0) > 0:
            return None

        latest_user_message = str(state.get("latest_user_message") or "").strip()
        if not latest_user_message:
            return None

        if AnswerNode._looks_like_road_identity_query(latest_user_message):
            content = (
                "暂未查询到该道路的编号或归属信息，无法直接确认。"
                "请提供更标准的道路名称、收费站名称或明确的道路编号后再查询。"
            )
        else:
            content = (
                "暂未查询到相关路况信息。"
                "请提供更标准的道路名称、收费站名称、方向或位置后再查询。"
            )
        return AIMessage(
            content=content,
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    @staticmethod
    def _looks_like_road_identity_query(message: str) -> bool:
        return any(
            keyword in message
            for keyword in (
                "编号",
                "路号",
                "哪条高速",
                "哪条路",
                "属于哪条高速",
                "在哪条高速",
                "哪条高速上",
                "是哪个高速上的",
                "是在哪条高速上",
            )
        )
