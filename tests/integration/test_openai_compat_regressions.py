"""真实回归问题的 OpenAI 兼容接口自动化测试。"""

from fastapi.testclient import TestClient


def _post_chat_completion(app_client: TestClient, question: str) -> object:
    return app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3535ba3b",
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "enable_thinking": False,
            "brief_answer": False,
        },
    )


def test_harvester_fee_query_uses_route_and_rag_and_keeps_policy_conclusion(
    regression_app_client: TestClient,
    regression_trace: dict[str, list[object]],
) -> None:
    response = _post_chat_completion(
        regression_app_client,
        "收割机从杭州到宁波要交多少费用",
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "免收" in content or "免费" in content
    assert "作业证" in content
    assert "64至92元" in content
    assert "普通车辆收费参考" in content

    live_agent_calls = regression_trace["live_agent_calls"]
    rag_queries = regression_trace["rag_queries"]
    assert [call["path"] for call in live_agent_calls] == ["/agent/driving"]
    assert len(rag_queries) >= 2
    assert any("免费通行" in query or "免收通行费" in query for query in rag_queries)
    assert any("收费政策" in query or "收费标准" in query for query in rag_queries)


def test_od_congestion_query_stays_route_only(
    regression_app_client: TestClient,
    regression_trace: dict[str, list[object]],
) -> None:
    response = _post_chat_completion(
        regression_app_client,
        "杭州到宁波堵不堵",
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "畅通" in content
    assert "拥堵" in content

    live_agent_calls = regression_trace["live_agent_calls"]
    assert [call["path"] for call in live_agent_calls] == ["/agent/driving"]
    assert regression_trace["rag_queries"] == []


def test_od_toll_query_stays_route_only_and_returns_ordinary_toll_reference(
    regression_app_client: TestClient,
    regression_trace: dict[str, list[object]],
) -> None:
    response = _post_chat_completion(
        regression_app_client,
        "我开车从杭州到宁波收费多少",
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "64至92元" in content
    assert "65元" in content
    assert "免费" not in content

    live_agent_calls = regression_trace["live_agent_calls"]
    assert [call["path"] for call in live_agent_calls] == ["/agent/driving"]
    assert regression_trace["rag_queries"] == []


def test_od_route_recommendation_stays_route_only(
    regression_app_client: TestClient,
    regression_trace: dict[str, list[object]],
) -> None:
    response = _post_chat_completion(
        regression_app_client,
        "杭州到宁波推荐一下路线",
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "推荐" in content
    assert "杭州湾环线高速" in content
    assert "2小时" in content

    live_agent_calls = regression_trace["live_agent_calls"]
    assert [call["path"] for call in live_agent_calls] == ["/agent/driving"]
    assert regression_trace["rag_queries"] == []


def test_agri_policy_query_uses_rag_and_route_together(
    regression_app_client: TestClient,
    regression_trace: dict[str, list[object]],
) -> None:
    response = _post_chat_completion(
        regression_app_client,
        "农机从绍兴到宁波免费通行有什么规定",
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "满足条件时可免费通行" in content
    assert "作业证" in content or "预约" in content
    assert "普通路线收费" in content

    live_agent_calls = regression_trace["live_agent_calls"]
    rag_queries = regression_trace["rag_queries"]
    assert [call["path"] for call in live_agent_calls] == ["/agent/driving"]
    assert len(rag_queries) >= 2
    assert any("免费通行" in query or "免收通行费" in query for query in rag_queries)


def test_non_od_policy_scope_query_uses_rag_instead_of_general_clarification(
    regression_app_client: TestClient,
    regression_trace: dict[str, list[object]],
) -> None:
    response = _post_chat_completion(
        regression_app_client,
        "甲鱼能免费吗",
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "甲鱼不能" in content
    assert "鲜活农产品品种目录" in content
    assert "通行费" in content

    assert regression_trace["live_agent_calls"] == []
    assert regression_trace["rag_queries"] != []
    assert any("甲鱼" in query for query in regression_trace["rag_queries"])
