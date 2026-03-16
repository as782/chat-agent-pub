"""OpenAI 兼容接口集成测试。"""

from fastapi.testclient import TestClient


def test_openai_compat_chat_completions_returns_standard_response(
    app_client: TestClient,
) -> None:
    """验证兼容接口会返回 OpenAI Chat Completions 兼容结构。"""

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [
                {"role": "system", "content": "你是一个简洁助手。"},
                {"role": "user", "content": "你好"},
            ],
            "user": "demo-user",
        },
    )
    response_payload = response.json()

    assert response.status_code == 200
    assert response_payload["id"].startswith("chatcmpl-")
    assert response_payload["object"] == "chat.completion"
    assert response_payload["model"] == "qwen-compatible-model"
    assert response_payload["choices"][0]["message"]["role"] == "assistant"
    assert response_payload["choices"][0]["message"]["content"] == "测试模型回答：你好"
    assert response_payload["choices"][0]["finish_reason"] == "stop"
    assert response_payload["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }


def test_openai_compat_chat_completions_rejects_stream_mode(app_client: TestClient) -> None:
    """验证兼容接口当前会拒绝流式输出请求。"""

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-compatible-model",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "unsupported_feature"
