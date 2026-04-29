"""Policy scope regression tests.

These cases lock in the generic handling for questions that ask whether a
subject belongs to a policy scope, catalog, or eligibility range.
"""

from app.agent.planner import PlannerService
from app.agent.state import ResolvedArguments


def test_policy_scope_subject_extraction_for_negative_catalog_question() -> None:
    assert PlannerService._infer_policy_subject("鲜花可以走绿通吗") == "鲜花"
    assert PlannerService._infer_policy_query_type("鲜花可以走绿通吗") == "policy_scope_check"
    assert PlannerService._infer_policy_keywords("鲜花可以走绿通吗") == [
        "鲜花",
        "鲜花 绿通",
        "鲜花 鲜活农产品",
        "鲜花 鲜活农产品目录",
    ]


def test_policy_scope_subject_extraction_for_positive_catalog_question() -> None:
    assert PlannerService._infer_policy_subject("鲜玉米可以走绿通吗") == "鲜玉米"
    assert PlannerService._infer_policy_query_type("鲜玉米可以走绿通吗") == "policy_scope_check"
    assert PlannerService._infer_policy_keywords("鲜玉米可以走绿通吗") == [
        "鲜玉米",
        "鲜玉米 绿通",
        "鲜玉米 鲜活农产品",
        "鲜玉米 鲜活农产品目录",
    ]


def test_policy_scope_subject_extraction_for_still_can_question() -> None:
    assert PlannerService._infer_policy_subject("鲜活农产品和目录外其他农产品混装还能走绿通吗") == (
        "鲜活农产品和目录外其他农产品混装"
    )
    assert (
        PlannerService._infer_policy_query_type("鲜活农产品和目录外其他农产品混装还能走绿通吗")
        == "policy_scope_check"
    )


def test_policy_scope_subject_extraction_for_processing_boundary_question() -> None:
    assert PlannerService._infer_policy_subject("去皮去叶清洗分割后的蔬菜还能走绿通吗") == (
        "去皮去叶清洗分割后的蔬菜"
    )
    assert PlannerService._infer_policy_query_type("去皮去叶清洗分割后的蔬菜还能走绿通吗") == (
        "policy_scope_check"
    )


def test_planner_enrich_step_metadata_keeps_scope_check_metadata_when_llm_is_broad() -> None:
    planner = PlannerService(llm_client=None)

    metadata = planner._enrich_step_metadata(
        executor="rag",
        metadata={"query_type": "policy_interpretation", "keywords": ["绿通"]},
        latest_user_message="鲜花可以走绿通吗",
        primary_category="policy",
    )

    assert metadata["query_type"] == "policy_scope_check"
    assert metadata["subject"] == "鲜花"
    assert metadata["keywords"] == [
        "鲜花",
        "鲜花 绿通",
        "鲜花 鲜活农产品",
        "鲜花 鲜活农产品目录",
    ]


def test_policy_scope_resolved_arguments_are_subject_preserving() -> None:
    arguments = ResolvedArguments(
        category="policy",
        arguments={
            "query": "鲜花可以走绿通吗",
            "query_type": "policy_scope_check",
            "subject": "鲜花",
            "keywords": ["鲜花", "鲜花 绿通", "鲜花 鲜活农产品", "鲜花 鲜活农产品目录"],
        },
    )

    assert arguments.arguments["subject"] == "鲜花"
    assert arguments.arguments["query_type"] == "policy_scope_check"


def test_policy_scope_supports_free_eligibility_subject_questions() -> None:
    assert PlannerService._infer_policy_query_type("\u7532\u9c7c\u80fd\u514d\u8d39\u5417") == "policy_scope_check"
    assert PlannerService._infer_policy_subject("\u7532\u9c7c\u80fd\u514d\u8d39\u5417") == "\u7532\u9c7c"


def test_policy_scope_does_not_hijack_non_green_channel_free_questions() -> None:
    assert PlannerService._infer_policy_query_type("\u9ad8\u901f\u62d6\u8f66\u514d\u8d39\u5417") != "policy_scope_check"
