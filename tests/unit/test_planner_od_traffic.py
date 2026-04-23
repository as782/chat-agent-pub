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


async def test_planner_routes_od_traffic_queries_to_route_only_plan() -> None:
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
                  "goal": "查路线",
                  "depends_on": [],
                  "can_run_in_parallel": false,
                  "metadata": {
                    "origin": "宁波",
                    "destination": "杭州",
                    "travel_mode": "driving",
                    "query": "宁波到杭州路况如何",
                    "query_intent": "route_planning"
                  }
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

    plan = await planner.build_plan_async(AgentState(latest_user_message="宁波到杭州路况如何？"))

    assert plan.primary_category == "route_planning"
    assert plan.execution_mode == "single_step"
    assert plan.recommended_route == "route"
    assert [step.step_id for step in plan.steps] == ["route_1", "answer_1"]
    assert [step.executor for step in plan.steps] == ["route", "answer"]
    assert plan.steps[0].metadata["origin"] == "宁波"
    assert plan.steps[0].metadata["destination"] == "杭州"
    assert plan.steps[0].metadata["query_intent"] == "route_planning"
    assert plan.steps[1].depends_on == ["route_1"]


def test_planner_traffic_status_detection_uses_stronger_intent() -> None:
    assert PlannerService._looks_like_traffic_status_query("G25今天堵不堵")
    assert not PlannerService._looks_like_traffic_status_query("浙江")
