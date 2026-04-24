"""回答节点单元测试。"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.nodes.answer_node import AnswerNode
from app.agent.answer_prompts import (
    COMPOSITE_ANSWER_PROMPT,
    NETWORK_REPORT_SUMMARY_PROMPT,
    ROUTE_SUMMARY_PROMPT,
    TRAFFIC_SUMMARY_PROMPT,
)
from app.agent.state import ExecutionPlan, ExecutionStep, ExecutorResult, PreparedContext
from app.clients.llm_client import LlmInputMessage
from app.persistence.base import Base
from app.persistence.message_repo import MessageRepository
from app.tools.registry import ExecutedToolCall


def test_answer_node_builds_executor_results_context() -> None:
    """统一 step_results 应被整理成可注入模型的上下文文本。"""

    context = AnswerNode._build_executor_results_context(
        {
            "rag_1": ExecutorResult(
                step_id="rag_1",
                executor="rag",
                is_success=True,
                normalized_result={"result_count": 2, "sources": ["doc-1", "doc-2"]},
                summary="知识检索命中 2 条结果。",
            ),
            "report_1": ExecutorResult(
                step_id="report_1",
                executor="report",
                is_success=True,
                normalized_result={"query_time": "2026-03-31 09:00:00", "congestion_total_mile": 12.5},
                summary="已整理路网报告任务参数。",
            ),
        }
    )

    assert context is not None
    assert "[rag_1] executor=rag success=True" in context
    assert "知识检索命中 2 条结果。" in context
    assert "12.5" in context


def test_answer_node_compacts_redundant_executor_result_fields() -> None:
    """结果上下文应保留关键信息，但去掉大块重复明细。"""

    context = AnswerNode._build_executor_results_context(
        {
            "traffic_1": ExecutorResult(
                step_id="traffic_1",
                executor="traffic",
                is_success=True,
                normalized_result={
                    "road_name": "沪昆高速",
                    "route_count": 2,
                    "event_count": 5,
                    "event_items": [{"event_category": "congestion"}],
                    "route_summaries": [{"route_index": 1}],
                },
                summary="路况查询成功，命中 1 条道路结果。",
            )
        }
    )

    assert context is not None
    assert "路况查询成功" in context
    assert "road_name" in context
    assert "route_count" in context
    assert "event_count" in context
    assert "event_items" not in context
    assert "route_summaries" not in context


def test_answer_node_resolves_prompt_name_from_category() -> None:
    assert (
        AnswerNode._resolve_answer_prompt_name({"primary_category": "traffic_status"})
        == "TRAFFIC_SUMMARY_PROMPT"
    )
    assert (
        AnswerNode._resolve_answer_prompt_name({"primary_category": "network_report"})
        == "NETWORK_REPORT_SUMMARY_PROMPT"
    )


def test_answer_node_uses_composite_prompt_for_route_congestion_questions() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "杭州到金华堵不堵",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
            "traffic_1": ExecutorResult(
                step_id="traffic_1",
                executor="traffic",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "COMPOSITE_ANSWER_PROMPT"


def test_answer_node_keeps_traffic_prompt_for_traffic_only_questions() -> None:
    state = {
        "primary_category": "traffic_status",
        "latest_user_message": "杭金衢高速堵不堵",
        "step_results": {
            "traffic_1": ExecutorResult(
                step_id="traffic_1",
                executor="traffic",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "TRAFFIC_SUMMARY_PROMPT"


def test_traffic_prompts_require_detailed_event_breakdown() -> None:
    assert "整体路况判断" in TRAFFIC_SUMMARY_PROMPT
    assert "拥堵情况" in TRAFFIC_SUMMARY_PROMPT
    assert "管制情况" in TRAFFIC_SUMMARY_PROMPT
    assert "事件情况" in TRAFFIC_SUMMARY_PROMPT
    assert "目前路况良好，通行基本正常" in TRAFFIC_SUMMARY_PROMPT
    assert "核心要求" in COMPOSITE_ANSWER_PROMPT
    assert "综合回答器" in COMPOSITE_ANSWER_PROMPT
    assert "用户最关心的结论" in COMPOSITE_ANSWER_PROMPT


def test_network_report_prompt_requires_strict_table_column_rules() -> None:
    assert (
        "| 序号 | 道路编号 | 高速名称 | 高速路段 | 收费站管控情况 | 路况 |"
        in NETWORK_REPORT_SUMMARY_PROMPT
    )
    assert "序号：只能输出阿拉伯数字" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "道路编号：只能输出类似 G25、G60 这种道路编号" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "高速名称：只输出高速名称本身，不带前置编号" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "高速路段：只输出具体路段信息" in NETWORK_REPORT_SUMMARY_PROMPT
    assert (
        "收费站管控情况：只写“收费站管制”汇总里的内容"
        in NETWORK_REPORT_SUMMARY_PROMPT
    )
    assert "写表前先按“收费站名称 + 道路方向”分组" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "每条记录必须包含“收费站名称 + 道路方向 + 出入口 + 管控结果”" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "同一收费站、同一道路方向下如果有多个出入口或多个管控结果，要合并成一行" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "不要把主线管制里的任何管制类型/措施写进这一列" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "路况：只写非收费站管控的情况，来源仅限主线管制和拥堵汇总" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "写表前先按“道路方向 + 具体路段”分组" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "多个独立拥堵/缓行点位仍然要逐条拆成多行" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "禁止把多个路段或多个缓行距离用分号、顿号合并到同一格里" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "返回的结果包含：查询时间、拥堵汇总、主线管制、收费站管制" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "拥堵汇总：每一行内容组成" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "事件分类, 管制措施, 现场情况, 占道情况, 开始时间, 预期结束时间, 事件描述" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "主线管制：每一行内容组成" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "收费站名称, 出入口" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "管制类型, 措施, 开始时间, 结束时间, 事件描述" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "收费站管制：每一行内容组成" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "管制状态, 措施, 开始时间, 结束时间, 事件描述" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "如果两者都没有，固定写“无”" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "主线管制写法必须包含“道路方向”" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "拥堵汇总写法组织为“道路方向 + 具体路段 + 事件描述 + 缓行距离”" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "没有明确路况时固定写“无”" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "当同一高速下存在多个收费站时，要优先按收费站拆成多行" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "多个独立拥堵/缓行点位仍然要逐条拆成多行" in NETWORK_REPORT_SUMMARY_PROMPT
    assert "写表前先分组：收费站按“收费站名称 + 道路方向”分组" in NETWORK_REPORT_SUMMARY_PROMPT


def test_answer_node_keeps_route_prompt_for_route_only_questions() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "杭州到金华怎么走",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "ROUTE_SUMMARY_PROMPT"


def test_route_summary_prompt_supports_dual_focus_modes() -> None:
    assert "默认以“路况与管制”为主" in ROUTE_SUMMARY_PROMPT
    assert "默认优先路况" in ROUTE_SUMMARY_PROMPT
    assert "出行方案" in ROUTE_SUMMARY_PROMPT
    assert "路况与管制" in ROUTE_SUMMARY_PROMPT
    assert "输出结构建议一" in ROUTE_SUMMARY_PROMPT
    assert "输出结构建议二" in ROUTE_SUMMARY_PROMPT
    assert "整体通行判断" in ROUTE_SUMMARY_PROMPT
    assert "{focus}" in ROUTE_SUMMARY_PROMPT


def test_answer_node_route_instruction_prioritizes_route_for_wayfinding() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "杭州到金华怎么走",
        "current_step_id": "answer_1",
        "execution_plan": ExecutionPlan(
            primary_category="route_planning",
            execution_mode="single_step",
            recommended_route="route",
            steps=[
                ExecutionStep(
                    step_id="route_1",
                    executor="route",
                    goal="查询起点到终点的推荐路线",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结路线结果并回答用户",
                    depends_on=["route_1"],
                    metadata={"focus": "出行方案"},
                ),
            ],
        ),
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
        },
    }

    instruction = AnswerNode._resolve_answer_instruction(state)

    assert "本轮回答焦点：路线推荐与关键路况" in instruction
    assert "推荐路线 -> 预计时长 -> 关键路况" in instruction
    assert "输出结构建议一" in instruction


def test_answer_node_route_instruction_defaults_to_traffic_for_generic_od() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "杭州到金华",
        "current_step_id": "answer_1",
        "execution_plan": ExecutionPlan(
            primary_category="route_planning",
            execution_mode="single_step",
            recommended_route="route",
            steps=[
                ExecutionStep(
                    step_id="route_1",
                    executor="route",
                    goal="查询起点到终点的推荐路线",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="总结路线结果并回答用户",
                    depends_on=["route_1"],
                    metadata={},
                ),
            ],
        ),
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
        },
    }

    instruction = AnswerNode._resolve_answer_instruction(state)

    assert "本轮回答焦点：路况与管制" in instruction
    assert "默认以“路况与管制”为主" in instruction


def test_answer_node_route_instruction_prioritizes_congestion_for_od_traffic() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "北京到上海堵吗",
        "current_step_id": "answer_1",
        "execution_plan": ExecutionPlan(
            primary_category="route_planning",
            execution_mode="single_step",
            recommended_route="route",
            steps=[
                ExecutionStep(
                    step_id="route_1",
                    executor="route",
                    goal="查询起点到终点的推荐路线",
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="根据路况查询结果回答用户关于北京到上海是否堵车的问题",
                    depends_on=["route_1"],
                    metadata={
                        "response_type": "text",
                        "focus": "出行方案",
                    },
                ),
            ],
        ),
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "ROUTE_SUMMARY_PROMPT"

    instruction = AnswerNode._resolve_answer_instruction(state)

    assert "本轮回答焦点：路况与管制" in instruction
    assert "优先给整体通行判断" in instruction
    assert "{focus}" not in instruction


def test_answer_node_uses_composite_prompt_for_route_and_rag_results() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "今天上高速到明天下高速要过路费吗",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
            "rag_1": ExecutorResult(
                step_id="rag_1",
                executor="rag",
                is_success=True,
            ),
        },
    }

    assert AnswerNode._resolve_answer_prompt_name(state) == "COMPOSITE_ANSWER_PROMPT"


def test_answer_node_builds_toll_focused_composite_instruction() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "今天上高速到明天下高速要过路费吗",
        "step_results": {
            "route_1": ExecutorResult(
                step_id="route_1",
                executor="route",
                is_success=True,
            ),
            "rag_1": ExecutorResult(
                step_id="rag_1",
                executor="rag",
                is_success=True,
            ),
        },
    }

    instruction = AnswerNode._resolve_answer_instruction(state)

    assert "收费判断" in instruction
    assert "rag、route" in instruction


@pytest.mark.asyncio
async def test_answer_node_reuses_tool_completion_result_without_new_llm_call(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """工具节点已得到完成结果时，回答节点应直接收口而不是再次调用模型。"""

    async def fail_create_chat_completion(*args, **kwargs) -> None:
        """如果进入普通回答分支，测试应直接失败。"""

        del args, kwargs
        raise AssertionError("answer_node 不应在已有 tool_completion_result 时再次调用 LLM")

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fail_create_chat_completion,
    )

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'answer-node.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-001",
                "prepared_context": PreparedContext(
                    messages=[],
                    used_session_memory=False,
                ),
                "tool_completion_result": AIMessage(
                    content="测试模型回答：工具结果是 2",
                    response_metadata={"finish_reason": "stop"},
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                ),
                "executed_tool_calls": [
                    ExecutedToolCall(
                        tool_call_id="call_calculator",
                        tool_name="calculator",
                        arguments={"expression": "1+1"},
                        output="2",
                    )
                ],
            }
        )

        persisted_messages = await MessageRepository(db_session).list_by_session("session-001")

        assert result["final_result"].content == "测试模型回答：工具结果是 2"
        assert result["final_result"].tool_calls[0].tool_name == "calculator"
        assert [message.role for message in persisted_messages] == ["assistant"]
        assert persisted_messages[0].content == "测试模型回答：工具结果是 2"

    await engine.dispose()

async def test_answer_node_regenerates_summary_for_multi_step_answer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """多步骤计划进入 answer 步时，应基于累计结果重新总结，而不是复用上一轮工具补全。"""

    async def fake_create_chat_completion(*args, **kwargs) -> AIMessage:
        """返回稳定的总结结果。"""

        del args, kwargs
        return AIMessage(
            content="测试模型回答：根据政策和路线结果生成统一总结。",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 16, "output_tokens": 12, "total_tokens": 28},
        )

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_chat_completion",
        fake_create_chat_completion,
    )

    def fake_create_runnable(self: object, **kwargs: object) -> object:
        """让回答节点通过 astream 获取稳定总结结果。"""

        del self, kwargs

        class _FakeRunnable:
            async def astream(self, messages: list[object], config=None):
                del messages, config
                ai_message = await fake_create_chat_completion()
                yield AIMessageChunk(
                    content=ai_message.content,
                    response_metadata=ai_message.response_metadata or {},
                    usage_metadata=ai_message.usage_metadata or {},
                    tool_call_chunks=[],
                )

        return _FakeRunnable()

    monkeypatch.setattr(
        "app.clients.llm_client.LlmClient.create_runnable",
        fake_create_runnable,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'answer-node-summary.db').as_posix()}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db_session:
        answer_node = AnswerNode(db_session)
        result = await answer_node.run(
            {
                "session_id": "session-002",
                "current_step_id": "answer_1",
                "execution_plan": ExecutionPlan(
                    primary_category="route_planning",
                    execution_mode="multi_step",
                    recommended_route="route",
                    steps=[
                        ExecutionStep(
                            step_id="rag_1",
                            executor="rag",
                            goal="检索政策标准",
                        ),
                        ExecutionStep(
                            step_id="route_1",
                            executor="route",
                            goal="查询路线方案",
                        ),
                        ExecutionStep(
                            step_id="answer_1",
                            executor="answer",
                            goal="汇总前置结果",
                            depends_on=["rag_1", "route_1"],
                        ),
                    ],
                ),
                "step_results": {
                    "rag_1": ExecutorResult(
                        step_id="rag_1",
                        executor="rag",
                        is_success=True,
                        normalized_result={"sources": ["policy-doc"]},
                    ),
                    "route_1": ExecutorResult(
                        step_id="route_1",
                        executor="route",
                        is_success=True,
                        normalized_result={"origin": "杭州", "destination": "金华"},
                    ),
                },
                "prepared_context": PreparedContext(
                    messages=[LlmInputMessage(role="system", content="请总结前置结果。")],
                    used_session_memory=False,
                ),
                "tool_completion_result": AIMessage(
                    content="测试模型回答：工具阶段临时结果。",
                    response_metadata={"finish_reason": "stop"},
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                ),
                "executed_tool_calls": [
                    ExecutedToolCall(
                        tool_call_id="call_route_plan",
                        tool_name="mcp_demo_http__route_plan",
                        arguments={"origin": "杭州", "destination": "金华"},
                        output="杭州到金华推荐走高速。",
                    )
                ],
            }
        )

        persisted_messages = await MessageRepository(db_session).list_by_session("session-002")

        assert result["final_result"].content == "测试模型回答：根据政策和路线结果生成统一总结。"
        assert result["final_result"].tool_calls[0].tool_name == "mcp_demo_http__route_plan"
        assert [message.role for message in persisted_messages] == ["assistant"]
        assert persisted_messages[0].content == "测试模型回答：根据政策和路线结果生成统一总结。"

    await engine.dispose()
