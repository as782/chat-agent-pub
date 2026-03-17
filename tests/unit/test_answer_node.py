"""回答节点单元测试。"""

from app.agent.nodes.answer_node import AnswerNode
from app.agent.state import ExecutorResult


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
                normalized_result={"scope": "全路网", "need_table": True},
                summary="已整理路网报告任务参数。",
            ),
        }
    )

    assert context is not None
    assert "[rag_1] executor=rag success=True" in context
    assert "知识检索命中 2 条结果。" in context
    assert "全路网" in context
