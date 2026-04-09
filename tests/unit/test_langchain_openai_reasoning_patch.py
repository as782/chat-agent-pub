"""Tests for the LangChain OpenAI reasoning-content compatibility patch."""

from __future__ import annotations

import importlib

from langchain_core.messages import AIMessage, AIMessageChunk


def test_llm_client_patches_langchain_openai_to_keep_reasoning_content() -> None:
    import app.clients.llm_client as llm_client_module
    import langchain_openai.chat_models.base as langchain_openai_base

    importlib.reload(llm_client_module)

    message = langchain_openai_base._convert_dict_to_message(
        {
            "role": "assistant",
            "content": "final answer",
            "reasoning_content": "thinking trace",
        }
    )

    assert isinstance(message, AIMessage)
    assert message.additional_kwargs["reasoning_content"] == "thinking trace"


def test_llm_client_patches_langchain_openai_stream_chunks_to_keep_reasoning_content() -> None:
    import app.clients.llm_client as llm_client_module
    import langchain_openai.chat_models.base as langchain_openai_base

    importlib.reload(llm_client_module)

    chunk = langchain_openai_base._convert_delta_to_message_chunk(
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "stream thinking",
        },
        AIMessageChunk,
    )

    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["reasoning_content"] == "stream thinking"
