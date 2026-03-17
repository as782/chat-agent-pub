"""端到端对话流程测试模块。
负责覆盖创建会话、多轮消息、知识库调用和历史查询等完整业务链路。
当前阶段使用测试桩隔离外部依赖，不负责真实第三方服务联调。
"""

from fastapi.testclient import TestClient

from app.schemas.knowledge import KnowledgeSearchResult


def test_e2e_create_session_then_chat_and_query_history(app_client: TestClient) -> None:
    """验证显式创建会话后，可以在同一会话中完成多轮对话并查询历史。"""

    create_session_response = app_client.post(
        "/api/v1/sessions",
        json={"title": "端到端测试会话", "user_id": "e2e-user"},
    )
    session_payload = create_session_response.json()
    session_id = session_payload["session_id"]

    first_chat_response = app_client.post(
        "/api/v1/chat",
        headers={"X-Session-ID": session_id},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我叫小王，请记住这个名字。"}],
        },
    )
    second_chat_response = app_client.post(
        "/api/v1/chat",
        headers={"X-Session-ID": session_id},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "我刚刚告诉你的名字是什么？"}],
        },
    )
    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert create_session_response.status_code == 201
    assert session_payload["title"] == "端到端测试会话"
    assert first_chat_response.status_code == 200
    assert second_chat_response.status_code == 200
    assert (
        second_chat_response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：你刚刚说你叫小王"
    )
    assert history_response.status_code == 200
    assert history_payload["total"] == 4
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_e2e_routes_to_knowledge_flow(app_client: TestClient, monkeypatch) -> None:
    """验证知识库前缀会触发知识库增强回答链路。"""

    async def fake_retrieve_for_agent(
        self: object,
        *,
        query: str,
        top_k: int = 4,
    ) -> list[KnowledgeSearchResult]:
        """返回稳定的知识库检索结果。"""

        del self, top_k
        assert query == "西湖在哪里？"
        return [
            KnowledgeSearchResult(
                document_id="doc-001",
                chunk_id="chunk-001",
                score=0.99,
                content="西湖位于杭州。",
                source="杭州百科",
            )
        ]

    monkeypatch.setattr(
        "app.agent.nodes.ragflow_node.KnowledgeService.retrieve_for_agent",
        fake_retrieve_for_agent,
    )

    response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "knowledge: 西湖在哪里？"}],
        },
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == "测试模型回答：根据知识库，西湖位于杭州。"
    )


def test_e2e_queries_history_after_tool_execution(app_client: TestClient) -> None:
    """验证工具调用完成后，可以通过历史消息接口看到完整执行轨迹。"""

    chat_response = app_client.post(
        "/api/v1/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "请帮我计算 1+1"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                }
            ],
        },
    )
    session_id = chat_response.headers["X-Session-ID"]
    history_response = app_client.get(f"/api/v1/messages/{session_id}")
    history_payload = history_response.json()

    assert chat_response.status_code == 200
    assert history_response.status_code == 200
    assert [message["role"] for message in history_payload["items"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history_payload["items"][2]["content"] == "2"
