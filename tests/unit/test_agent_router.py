"""Agent 路由决策单元测试。"""

from app.agent.router import resolve_agent_route


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


def test_router_marks_mcp_requests() -> None:
    """验证 MCP 前缀会被标记为 MCP 路由。"""

    route = resolve_agent_route({"latest_user_message": "mcp: 查询远端工具"})

    assert route == "mcp"


def test_router_defaults_to_answer_route() -> None:
    """验证普通问答请求会回落到默认回答路由。"""

    route = resolve_agent_route({"latest_user_message": "你好"})

    assert route == "answer"
