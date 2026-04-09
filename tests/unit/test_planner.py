"""Planner 服务单元测试。"""

import logging

from langchain_core.messages import AIMessage

from app.agent.planner import PlannerService
from app.agent.state import AgentState
from app.core.config import Settings


class _FakePlannerLlmClient:
    """最小 LLM planner 测试桩。"""

    def __init__(self, content: str) -> None:
        self._content = content
        self.last_kwargs: dict[str, object] | None = None

    async def create_chat_completion(self, **_: object) -> AIMessage:
        self.last_kwargs = _
        if self._content == "error":
            raise RuntimeError("LLM Error")
        return AIMessage(
            content=self._content,
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )


async def test_planner_can_use_llm_output() -> None:
    """验证 LLM planner 能够正确解析 LLM 返回的 JSON。"""

    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "route_planning",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "rag_1",
                  "executor": "rag",
                  "goal": "检索政策要求",
                  "depends_on": [],
                  "can_run_in_parallel": true,
                  "metadata": {}
                },
                {
                  "step_id": "route_1",
                  "executor": "route",
                  "goal": "查询路线规划",
                  "depends_on": [],
                  "can_run_in_parallel": true,
                  "metadata": {}
                },
                {
                  "step_id": "answer_1",
                  "executor": "answer",
                  "goal": "汇总回答",
                  "depends_on": ["rag_1", "route_1"],
                  "can_run_in_parallel": false,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    plan = await planner.build_plan_async(
        AgentState(latest_user_message="杭州到金华怎么走，并说明是否符合高速清障标准？")
    )

    assert plan.primary_category == "route_planning"
    assert plan.execution_mode == "multi_step"
    assert plan.recommended_route == "ragflow"
    assert [step.executor for step in plan.steps] == ["rag", "route", "answer"]


async def test_planner_falls_back_when_llm_raises_error() -> None:
    """LLM 客户端抛出异常时，应使用兜底规则。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(AgentState(latest_user_message="发生错误"))

    assert plan.primary_category == "general"
    assert plan.recommended_route == "answer"
    assert [step.executor for step in plan.steps] == ["answer"]


async def test_planner_prefers_dedicated_base_url(monkeypatch) -> None:
    """Planner should use its dedicated base URL when configured."""

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("PLANNER_BASE_URL", "https://planner.example.com/v1")
    monkeypatch.setenv("PLANNER_MODEL", "planner-model")
    fake_llm_client = _FakePlannerLlmClient(
        """
        {
          "primary_category": "general",
          "need_clarification": false,
          "clarification_question": null,
          "steps": [
            {
              "step_id": "answer_1",
              "executor": "answer",
              "goal": "Directly answer the user",
              "depends_on": [],
              "can_run_in_parallel": false,
              "metadata": {}
            }
          ]
        }
        """
    )
    planner = PlannerService(llm_client=fake_llm_client)

    await planner.build_plan_async(AgentState(latest_user_message="hello"))

    assert fake_llm_client.last_kwargs is not None
    assert fake_llm_client.last_kwargs["model_name"] == "planner-model"
    assert fake_llm_client.last_kwargs["base_url"] == "https://planner.example.com/v1"


async def test_planner_prefers_dedicated_api_key(monkeypatch) -> None:
    """Planner should use its dedicated API key when configured."""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("PLANNER_API_KEY", "planner-test-key")
    fake_llm_client = _FakePlannerLlmClient(
        """
        {
          "primary_category": "general",
          "need_clarification": false,
          "clarification_question": null,
          "steps": [
            {
              "step_id": "answer_1",
              "executor": "answer",
              "goal": "Directly answer the user",
              "depends_on": [],
              "can_run_in_parallel": false,
              "metadata": {}
            }
          ]
        }
        """
    )
    planner = PlannerService(llm_client=fake_llm_client)

    await planner.build_plan_async(AgentState(latest_user_message="hello"))

    assert fake_llm_client.last_kwargs is not None
    assert fake_llm_client.last_kwargs["api_key"] == "planner-test-key"


async def test_planner_prefers_dedicated_timeout(monkeypatch) -> None:
    """Planner should use its dedicated timeout when configured."""

    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("PLANNER_TIMEOUT_SECONDS", "15")
    fake_llm_client = _FakePlannerLlmClient(
        """
        {
          "primary_category": "general",
          "need_clarification": false,
          "clarification_question": null,
          "steps": [
            {
              "step_id": "answer_1",
              "executor": "answer",
              "goal": "Directly answer the user",
              "depends_on": [],
              "can_run_in_parallel": false,
              "metadata": {}
            }
          ]
        }
        """
    )
    planner = PlannerService(llm_client=fake_llm_client)

    await planner.build_plan_async(AgentState(latest_user_message="hello"))

    assert fake_llm_client.last_kwargs is not None
    assert fake_llm_client.last_kwargs["timeout_seconds"] == 15.0


async def test_planner_uses_configured_thinking_flag(monkeypatch) -> None:
    """Planner should respect its dedicated thinking toggle when configured."""

    monkeypatch.setenv("PLANNER_ENABLE_THINKING", "true")
    fake_llm_client = _FakePlannerLlmClient(
        """
        {
          "primary_category": "general",
          "need_clarification": false,
          "clarification_question": null,
          "steps": [
            {
              "step_id": "answer_1",
              "executor": "answer",
              "goal": "Directly answer the user",
              "depends_on": [],
              "can_run_in_parallel": false,
              "metadata": {}
            }
          ]
        }
        """
    )
    planner = PlannerService(llm_client=fake_llm_client)

    await planner.build_plan_async(AgentState(latest_user_message="hello"))

    assert fake_llm_client.last_kwargs is not None
    assert fake_llm_client.last_kwargs["enable_thinking"] is True


async def test_planner_falls_back_to_main_api_key_when_dedicated_key_blank(monkeypatch) -> None:
    """Planner should fall back to main API key when dedicated key is blank."""

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("PLANNER_API_KEY", "   ")
    fake_llm_client = _FakePlannerLlmClient(
        """
        {
          "primary_category": "general",
          "need_clarification": false,
          "clarification_question": null,
          "steps": [
            {
              "step_id": "answer_1",
              "executor": "answer",
              "goal": "Directly answer the user",
              "depends_on": [],
              "can_run_in_parallel": false,
              "metadata": {}
            }
          ]
        }
        """
    )
    planner = PlannerService(llm_client=fake_llm_client)

    await planner.build_plan_async(AgentState(latest_user_message="hello"))

    assert fake_llm_client.last_kwargs is not None
    assert fake_llm_client.last_kwargs["api_key"] is None


async def test_planner_falls_back_to_main_timeout_when_dedicated_timeout_missing(monkeypatch) -> None:
    """Planner should fall back to main timeout when dedicated timeout is missing."""

    fake_llm_client = _FakePlannerLlmClient(
        """
        {
          "primary_category": "general",
          "need_clarification": false,
          "clarification_question": null,
          "steps": [
            {
              "step_id": "answer_1",
              "executor": "answer",
              "goal": "Directly answer the user",
              "depends_on": [],
              "can_run_in_parallel": false,
              "metadata": {}
            }
          ]
        }
        """
    )
    planner = PlannerService(
        llm_client=fake_llm_client,
        settings=Settings.model_construct(
            openai_timeout_seconds=60.0,
            openai_enable_thinking=None,
            planner_timeout_seconds=None,
            planner_api_key=None,
            planner_base_url=None,
            planner_model=None,
            planner_enable_thinking=None,
        ),
    )

    await planner.build_plan_async(AgentState(latest_user_message="hello"))

    assert fake_llm_client.last_kwargs is not None
    assert fake_llm_client.last_kwargs["timeout_seconds"] == 60.0


async def test_planner_fallback_handles_explicit_tools() -> None:
    """当发生兜底且用户显式请求工具时，应生成包含 tool 的保底计划。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("not-json"))

    plan = await planner.build_plan_async(
        AgentState(latest_user_message="使用计算器", requested_tool_names=["calculator"])
    )

    assert plan.primary_category == "general"
    assert plan.recommended_route == "tool"
    assert [step.executor for step in plan.steps] == ["tool", "answer"]


async def test_planner_fallback_can_classify_service_area() -> None:
    """当 LLM planner 失败时，应能识别服务区查询问题。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(
        AgentState(latest_user_message="杭州东服务区充电桩情况怎么样？")
    )

    assert plan.primary_category == "service_area"
    assert plan.recommended_route == "service"
    assert [step.executor for step in plan.steps] == ["service", "answer"]


async def test_planner_normalizes_province_wide_traffic_to_network_report() -> None:
    """全省整体路况类问题应纠偏到 network_report。"""

    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "traffic_status",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "traffic_1",
                  "executor": "traffic",
                  "goal": "查询路况",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {}
                },
                {
                  "step_id": "answer_1",
                  "executor": "answer",
                  "goal": "输出结果",
                  "depends_on": ["traffic_1"],
                  "can_run_in_parallel": false,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    plan = await planner.build_plan_async(
        AgentState(latest_user_message="请提供浙江省内高速实时路况")
    )

    assert plan.primary_category == "traffic_status"
    assert plan.recommended_route == "traffic"
    assert [step.executor for step in plan.steps] == ["traffic", "answer"]


async def test_planner_fallback_normalizes_province_wide_traffic_to_network_report() -> None:
    """兜底规则下也应把全省整体路况问题归入 network_report。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(
        AgentState(latest_user_message="请对比给出浙江省内整体高速路况")
    )

    assert plan.primary_category == "network_report"
    assert plan.recommended_route == "report"
    assert [step.executor for step in plan.steps] == ["report", "answer"]


async def test_planner_routes_current_time_to_builtin_tool() -> None:
    """当前时间类问题应保持 general 分类，但走 tool -> answer。"""

    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "general",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "answer_1",
                  "executor": "answer",
                  "goal": "直接回答用户",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    plan = await planner.build_plan_async(AgentState(latest_user_message="当前时间"))

    assert plan.primary_category == "general"
    assert plan.recommended_route == "answer"
    assert [step.executor for step in plan.steps] == ["answer"]


async def test_planner_routes_calculation_to_builtin_tool() -> None:
    """简单计算问题应保持 general 分类，但走 tool -> answer。"""

    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "general",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "answer_1",
                  "executor": "answer",
                  "goal": "直接回答用户",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    plan = await planner.build_plan_async(AgentState(latest_user_message="1+2"))

    assert plan.primary_category == "general"
    assert plan.recommended_route == "answer"
    assert [step.executor for step in plan.steps] == ["answer"]


async def test_planner_routes_expression_with_chinese_symbols_to_builtin_tool() -> None:
    """带中文破折号和问号的纯算式也应走计算工具。"""

    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "general",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "answer_1",
                  "executor": "answer",
                  "goal": "直接回答用户",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    plan = await planner.build_plan_async(
        AgentState(latest_user_message="120—4+44*73*12=？")
    )

    assert plan.primary_category == "general"
    assert plan.recommended_route == "answer"
    assert [step.executor for step in plan.steps] == ["answer"]


async def test_planner_logs_llm_response_content(caplog) -> None:
    """Planner should log the raw LLM response content for debugging."""

    caplog.set_level(logging.INFO, logger="app.agent.planner")
    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "general",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "answer_1",
                  "executor": "answer",
                  "goal": "Directly answer the user",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    await planner.build_plan_async(AgentState(latest_user_message="hello"))

    assert "Planner LLM response received:" in caplog.text
    assert '"primary_category": "general"' in caplog.text


async def test_planner_logs_final_execution_plan(caplog) -> None:
    """Planner should log the normalized execution plan used for actual routing."""

    caplog.set_level(logging.INFO, logger="app.agent.planner")
    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    await planner.build_plan_async(AgentState(latest_user_message="杭州到金华堵不堵"))

    assert "Planner final execution plan:" in caplog.text
    assert '"primary_category": "traffic_status"' in caplog.text
    assert '"executor": "route"' in caplog.text
    assert '"executor": "traffic"' in caplog.text


async def test_planner_fallback_routes_direct_traffic_question_to_traffic() -> None:
    """单条道路路况问题应直接走 traffic -> answer。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(AgentState(latest_user_message="G25今天堵不堵"))

    assert plan.primary_category == "traffic_status"
    assert plan.recommended_route == "traffic"
    assert [step.executor for step in plan.steps] == ["traffic", "answer"]


async def test_planner_fallback_routes_policy_question_to_rag() -> None:
    """政策问题应直接走 rag -> answer。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(AgentState(latest_user_message="绿通政策是什么"))

    assert plan.primary_category == "policy"
    assert plan.recommended_route == "ragflow"
    assert [step.executor for step in plan.steps] == ["rag", "answer"]


async def test_planner_fallback_routes_network_report_question_to_report() -> None:
    """全省路况对比报表问题应直接走 report -> answer。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(AgentState(latest_user_message="浙江全省今天路况对比报表"))

    assert plan.primary_category == "network_report"
    assert plan.recommended_route == "report"
    assert [step.executor for step in plan.steps] == ["report", "answer"]


async def test_planner_fallback_routes_od_congestion_to_route_then_traffic() -> None:
    """OD + 拥堵问题应走 route -> traffic -> answer。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(AgentState(latest_user_message="杭州到金华堵不堵"))

    assert plan.primary_category == "traffic_status"
    assert plan.recommended_route == "route"
    assert [step.executor for step in plan.steps] == ["route", "traffic", "answer"]


async def test_planner_keeps_llm_route_plan_when_it_is_valid() -> None:
    """当 LLM 已经给出有效步骤时，应保留其原始计划，只做合法性补齐。"""

    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "route_planning",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "route_1",
                  "executor": "route",
                  "goal": "查询路线",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {}
                },
                {
                  "step_id": "answer_1",
                  "executor": "answer",
                  "goal": "总结结果",
                  "depends_on": ["route_1"],
                  "can_run_in_parallel": false,
                  "metadata": {}
                }
              ]
            }
            """
        )
    )

    plan = await planner.build_plan_async(AgentState(latest_user_message="杭州到金华堵不堵"))

    assert plan.primary_category == "route_planning"
    assert plan.recommended_route == "route"
    assert [step.executor for step in plan.steps] == ["route", "answer"]
