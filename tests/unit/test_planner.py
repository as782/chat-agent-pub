"""Planner 服务单元测试。"""

from langchain_core.messages import AIMessage

from app.agent.planner import PlannerService
from app.agent.state import AgentState


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
