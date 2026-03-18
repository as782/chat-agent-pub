"""Planner 服务单元测试。"""

from app.agent.planner import PlannerService
from app.agent.state import AgentState
from langchain_core.messages import AIMessage


class _FakePlannerLlmClient:
    """最小 LLM planner 测试桩。"""

    def __init__(self, content: str) -> None:
        self._content = content

    async def create_chat_completion(self, **_: object) -> AIMessage:
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


async def test_planner_falls_back_when_llm_output_is_invalid() -> None:
    """LLM planner 解析失败时，应使用兜底规则。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("not-json"))

    plan = await planner.build_plan_async(AgentState(latest_user_message="杭州到金华怎么走"))

    assert plan.primary_category == "general"
    assert plan.recommended_route == "answer"
    assert [step.executor for step in plan.steps] == ["answer"]


async def test_planner_falls_back_when_llm_raises_error() -> None:
    """LLM 客户端抛出异常时，应使用兜底规则。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("error"))

    plan = await planner.build_plan_async(AgentState(latest_user_message="发生错误"))

    assert plan.primary_category == "general"
    assert plan.recommended_route == "answer"
    assert [step.executor for step in plan.steps] == ["answer"]


async def test_planner_fallback_handles_explicit_tools() -> None:
    """当发生兜底且用户显式请求工具时，应生成包含 tool 的保底计划。"""

    planner = PlannerService(llm_client=_FakePlannerLlmClient("not-json"))

    plan = await planner.build_plan_async(
        AgentState(
            latest_user_message="使用计算器",
            requested_tool_names=["calculator"]
        )
    )

    assert plan.primary_category == "general"
    assert plan.recommended_route == "tool"
    assert [step.executor for step in plan.steps] == ["tool", "answer"]
