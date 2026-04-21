from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.planner import PlannerService
from app.agent.state import AgentState


class _FakePlannerLlmClient:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create_chat_completion(self, **_: object) -> AIMessage:
        return AIMessage(
            content=self._content,
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )


async def test_planner_preserves_valid_llm_route_then_traffic_steps_after_category_normalization() -> None:
    planner = PlannerService(
        llm_client=_FakePlannerLlmClient(
            """
            {
              "primary_category": "route_planning",
              "need_clarification": false,
              "clarification_question": null,
              "steps": [
                {
                  "step_id": "1",
                  "executor": "route",
                  "goal": "规划从宁波到杭州的推荐行驶路线",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {
                    "origin": "宁波",
                    "destination": "杭州",
                    "travel_mode": "driving",
                    "query": "宁波到杭州路况如何"
                  }
                },
                {
                  "step_id": "2",
                  "executor": "traffic",
                  "goal": "查询规划路线涉及路段的实时路况",
                  "depends_on": ["1"],
                  "can_run_in_parallel": true,
                  "metadata": {
                    "query": "宁波到杭州路况",
                    "roads": "通过步骤1获取的具体高速/道路编号及名称列表",
                    "target": "宁波至杭州全程",
                    "query_intent": "route_based_traffic"
                  }
                },
                {
                  "step_id": "3",
                  "executor": "answer",
                  "goal": "汇总路线和路况信息",
                  "depends_on": ["1", "2"],
                  "can_run_in_parallel": false,
                  "metadata": {
                    "focus": "路线推荐、拥堵程度"
                  }
                }
              ]
            }
            """
        )
    )

    plan = await planner.build_plan_async(AgentState(latest_user_message="宁波到杭州路况如何？"))

    assert plan.primary_category == "traffic_status"
    assert plan.execution_mode == "multi_step"
    assert plan.recommended_route == "route"
    assert [step.step_id for step in plan.steps] == ["1", "2", "3"]
    assert [step.executor for step in plan.steps] == ["route", "traffic", "answer"]
    assert plan.steps[1].depends_on == ["1"]
    assert plan.steps[1].metadata["target"] == "宁波至杭州全程"
    assert plan.steps[2].depends_on == ["1", "2"]
