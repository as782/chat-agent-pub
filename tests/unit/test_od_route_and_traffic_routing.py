from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.argument_resolver import ArgumentResolver
from app.agent.answer_prompts import COMPOSITE_ANSWER_PROMPT, TRAFFIC_SUMMARY_PROMPT
from app.agent.nodes.answer_node import AnswerNode
from app.agent.planner import PlannerService
from app.agent.state import AgentState, ExecutionPlan, ExecutionStep, ExecutorResult


class _FailingPlannerLlmClient:
    async def create_chat_completion(self, **_: object) -> AIMessage:
        raise RuntimeError("force fallback")


async def test_explicit_route_queries_use_route_only_plan() -> None:
    planner = PlannerService(llm_client=_FailingPlannerLlmClient())

    queries = [
        "从杭州去舟山怎么走不堵？",
        "从宁波到温州走哪条路最快？",
        "长沙到杭州萧山那条高速不堵吗？",
        "上海到台州，马上出发，推荐一下路线，谢谢",
        "深圳到湖北怎么开？",
    ]

    for query in queries:
        plan = await planner.build_plan_async(AgentState(latest_user_message=query))
        assert plan.primary_category == "route_planning"
        assert plan.execution_mode == "single_step"
        assert plan.recommended_route == "route"
        assert [step.executor for step in plan.steps] == ["route", "answer"]
        assert plan.steps[1].depends_on == ["route_1"]


async def test_od_traffic_queries_default_to_route_only_plan() -> None:
    planner = PlannerService(llm_client=_FailingPlannerLlmClient())

    queries = [
        "我从衢州去宁波？",
        "上饶到温州堵车吗？",
        "安徽到浙江堵吗？",
        "婺源到慈溪怎么样？",
        "杭州收费站到金华正常通行吗？",
        "温州回杭州堵吗？",
    ]

    for query in queries:
        plan = await planner.build_plan_async(AgentState(latest_user_message=query))
        assert plan.primary_category == "route_planning"
        assert plan.execution_mode == "single_step"
        assert plan.recommended_route == "route"
        assert [step.executor for step in plan.steps] == ["route", "answer"]
        assert plan.steps[1].depends_on == ["route_1"]


async def test_direct_traffic_queries_use_traffic_only_plan() -> None:
    planner = PlannerService(llm_client=_FailingPlannerLlmClient())

    queries = [
        "衢州东那边能看一下吗？",
        "沪武高速堵不堵？",
        "沪武高速现在怎么样？",
        "白果收费站进口是正常的吗",
    ]

    for query in queries:
        plan = await planner.build_plan_async(AgentState(latest_user_message=query))
        assert plan.primary_category == "traffic_status"
        assert plan.execution_mode == "single_step"
        assert plan.recommended_route == "traffic"
        assert [step.executor for step in plan.steps] == ["traffic", "answer"]


def test_argument_resolver_cleans_colloquial_od_endpoints() -> None:
    resolver = ArgumentResolver()

    result = resolver.resolve(
        {
            "latest_user_message": "我从衢州去宁波？",
            "primary_category": "traffic_status",
        }
    )
    assert result.arguments["query_intent"] == "route_based_traffic"

    route_result = resolver.resolve(
        {
            "latest_user_message": "温州回杭州堵吗？",
            "primary_category": "route_planning",
        }
    )
    assert route_result.arguments["origin"] == "温州"
    assert route_result.arguments["destination"] == "杭州"


def test_answer_node_prefers_composite_route_and_traffic_prompt_structure() -> None:
    state = {
        "primary_category": "route_planning",
        "latest_user_message": "从杭州去舟山怎么走不堵？",
        "execution_plan": ExecutionPlan(
            primary_category="route_planning",
            execution_mode="multi_step",
            recommended_route="route",
            steps=[
                ExecutionStep(step_id="route_1", executor="route", goal="查询路线"),
                ExecutionStep(
                    step_id="traffic_1",
                    executor="traffic",
                    goal="查询路况",
                    depends_on=["route_1"],
                ),
                ExecutionStep(
                    step_id="answer_1",
                    executor="answer",
                    goal="输出答案",
                    depends_on=["route_1", "traffic_1"],
                ),
            ],
        ),
        "step_results": {
            "route_1": ExecutorResult(step_id="route_1", executor="route", is_success=True),
            "traffic_1": ExecutorResult(step_id="traffic_1", executor="traffic", is_success=True),
        },
    }

    instruction = AnswerNode._resolve_answer_instruction(state)
    assert AnswerNode._resolve_answer_prompt_name(state) == "COMPOSITE_ANSWER_PROMPT"
    assert "推荐路线" in instruction
    assert "预计时长" in COMPOSITE_ANSWER_PROMPT
    assert "综合回答器" in COMPOSITE_ANSWER_PROMPT
    assert "整体路况判断" in TRAFFIC_SUMMARY_PROMPT
