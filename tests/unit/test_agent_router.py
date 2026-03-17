"""Agent 路由决策单元测试。"""

from app.agent.router import resolve_agent_route
from app.agent.state import ExecutionPlan


def test_router_prefers_tool_route_when_tools_are_requested() -> None:
    """验证请求显式携带工具定义时会优先标记为工具路由。"""

    route = resolve_agent_route(
        {
            "latest_user_message": "请帮我计算 1+1",
            "requested_tool_names": ["calculator"],
        }
    )

    assert route == "tool"


def test_router_marks_knowledge_requests() -> None:
    """验证知识库前缀会被标记为知识库路由。"""

    route = resolve_agent_route({"latest_user_message": "知识库: 帮我查公司制度"})

    assert route == "ragflow"


def test_router_marks_english_knowledge_requests() -> None:
    """验证英文 knowledge 前缀也会命中知识库路由。"""

    route = resolve_agent_route({"latest_user_message": "knowledge: 高速清障最低标准是什么？"})

    assert route == "ragflow"


def test_router_marks_common_misspelled_knowledge_requests() -> None:
    """验证常见拼错的 konwledge 前缀也会命中知识库路由。"""

    route = resolve_agent_route({"latest_user_message": "konwledge: 高速清障最低标准是什么？"})

    assert route == "ragflow"


def test_router_marks_mcp_requests() -> None:
    """验证 MCP 前缀会被标记为 MCP 路由。"""

    route = resolve_agent_route({"latest_user_message": "mcp: 查询远端工具"})

    assert route == "mcp"


def test_router_defaults_to_answer_route() -> None:
    """验证普通问答请求会回落到默认回答路由。"""

    route = resolve_agent_route({"latest_user_message": "你好"})

    assert route == "answer"


def test_router_prefers_planner_recommended_route_when_available() -> None:
    """验证 planner 已输出推荐路由时，router 会优先消费该结果。"""

    route = resolve_agent_route(
        {
            "latest_user_message": "你好",
            "execution_plan": ExecutionPlan(
                primary_category="policy",
                execution_mode="single_step",
                recommended_route="ragflow",
            ),
        }
    )

    assert route == "ragflow"


def test_router_prefers_scheduled_route_over_explicit_tools() -> None:
    """进入计划执行过程后，应优先使用 scheduler 当前轮次给出的路由。"""

    route = resolve_agent_route(
        {
            "latest_user_message": "请帮我计算 1+1",
            "requested_tool_names": ["calculator"],
            "scheduled_route": "answer",
        }
    )

    assert route == "answer"


def test_router_still_prefers_explicit_tools_over_planner_route() -> None:
    """验证显式传入 tools 时，仍保持最高优先级。"""

    route = resolve_agent_route(
        {
            "latest_user_message": "你好",
            "requested_tool_names": ["calculator"],
            "execution_plan": ExecutionPlan(
                primary_category="policy",
                execution_mode="single_step",
                recommended_route="ragflow",
            ),
        }
    )

    assert route == "tool"
