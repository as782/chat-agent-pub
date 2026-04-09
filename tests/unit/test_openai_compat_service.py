"""Unit tests for the OpenAI compatibility service."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk

from app.agent.state import ChatTurnResult
from app.services.openai_compat_service import OpenAICompatService


def test_build_chat_completion_response_includes_reasoning_content() -> None:
    service = OpenAICompatService()

    response = service.build_chat_completion_response(
        ChatTurnResult(
            session_id="session-001",
            content="最终回答",
            model_name="qwen-compatible-model",
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            finish_reason="stop",
            route="answer",
            reasoning_content="这是模型的思考过程",
        )
    )

    assert response.choices[0].message.content == "最终回答"
    assert response.choices[0].message.reasoning_content == "这是模型的思考过程"


def test_stream_chunk_builder_includes_reasoning_content_delta() -> None:
    service = OpenAICompatService()
    builder = service.create_stream_chunk_builder(default_model_name="test-model")

    payloads = builder.consume_chunk(
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "第一段思考"},
            response_metadata={"model_name": "test-model"},
        )
    )

    assert len(payloads) == 1
    assert '"reasoning_content": "第一段思考"' in payloads[0]


def test_build_chat_completion_response_reads_reasoning_content_from_ai_message() -> None:
    service = OpenAICompatService()

    response = service.build_chat_completion_response(
        AIMessage(
            content="最终回答",
            additional_kwargs={"reasoning_content": "模型思考"},
            response_metadata={"model_name": "qwen-compatible-model", "finish_reason": "stop"},
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )
    )

    assert response.choices[0].message.reasoning_content == "模型思考"
