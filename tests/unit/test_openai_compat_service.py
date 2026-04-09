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
            content="final answer",
            model_name="qwen-compatible-model",
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            finish_reason="stop",
            route="answer",
            reasoning_content="thinking trace",
        )
    )

    assert response.choices[0].message.content == "<think>thinking trace</think>final answer"
    assert response.choices[0].message.reasoning_content == "thinking trace"


def test_stream_chunk_builder_includes_reasoning_content_delta_and_think_wrapper() -> None:
    service = OpenAICompatService()
    builder = service.create_stream_chunk_builder(default_model_name="test-model")

    payloads = builder.consume_chunk(
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "first thought"},
            response_metadata={"model_name": "test-model"},
        )
    )

    assert len(payloads) == 1
    assert '"reasoning_content": "first thought"' in payloads[0]
    assert '"content": "<think>first thought"' in payloads[0]


def test_stream_chunk_builder_closes_think_tag_before_answer_content() -> None:
    service = OpenAICompatService()
    builder = service.create_stream_chunk_builder(default_model_name="test-model")

    builder.consume_chunk(
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "first thought"},
            response_metadata={"model_name": "test-model"},
        )
    )
    payloads = builder.consume_chunk(
        AIMessageChunk(
            content="final answer",
            response_metadata={"model_name": "test-model"},
        )
    )

    assert len(payloads) == 1
    assert '"content": "</think>final answer"' in payloads[0]


def test_stream_chunk_builder_closes_think_tag_before_finish_when_reasoning_only() -> None:
    service = OpenAICompatService()
    builder = service.create_stream_chunk_builder(default_model_name="test-model")

    builder.consume_chunk(
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "first thought"},
            response_metadata={"model_name": "test-model"},
        )
    )
    payloads = builder.finalize()

    assert '"content": "</think>"' in payloads[0]
    assert '"finish_reason": "stop"' in payloads[1]


def test_build_chat_completion_response_reads_reasoning_content_from_ai_message() -> None:
    service = OpenAICompatService()

    response = service.build_chat_completion_response(
        AIMessage(
            content="final answer",
            additional_kwargs={"reasoning_content": "model thinking"},
            response_metadata={"model_name": "qwen-compatible-model", "finish_reason": "stop"},
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )
    )

    assert response.choices[0].message.content == "<think>model thinking</think>final answer"
    assert response.choices[0].message.reasoning_content == "model thinking"
