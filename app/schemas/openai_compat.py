"""OpenAI-compatible chat completion schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAITextContentPart(BaseModel):
    """Text content part."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["text"] = Field(default="text", description="Content part type.")
    text: str = Field(description="Text content.")


class OpenAIChatCompletionToolFunction(BaseModel):
    """Function tool definition."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Tool name.")
    description: str | None = Field(default=None, description="Tool description.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for tool parameters.",
    )


class OpenAIChatCompletionTool(BaseModel):
    """Tool definition."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = Field(default="function", description="Tool type.")
    function: OpenAIChatCompletionToolFunction = Field(description="Function tool payload.")


class OpenAIChatCompletionToolCallFunction(BaseModel):
    """Function payload inside a tool call."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Function name.")
    arguments: str = Field(description="JSON string of function arguments.")


class OpenAIChatCompletionToolCall(BaseModel):
    """Tool call payload."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Tool call identifier.")
    type: Literal["function"] = Field(default="function", description="Tool call type.")
    function: OpenAIChatCompletionToolCallFunction = Field(
        description="Function call payload.",
    )


class OpenAIChatMessage(BaseModel):
    """Chat message."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"] = Field(
        description="Message role.",
    )
    content: str | list[OpenAITextContentPart] | None = Field(
        default=None,
        description="Message content.",
    )
    name: str | None = Field(default=None, description="Optional message name.")
    tool_call_id: str | None = Field(default=None, description="Related tool call id.")
    tool_calls: list[OpenAIChatCompletionToolCall] | None = Field(
        default=None,
        description="Tool calls carried by the assistant message.",
    )


class OpenAIChatTemplateKwargs(BaseModel):
    """Nested chat_template_kwargs payload."""

    model_config = ConfigDict(extra="ignore")

    enable_thinking: bool | None = Field(
        default=None,
        description="Optional thinking toggle for compatible upstream gateways.",
    )


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(description="Requested model name.")
    messages: list[OpenAIChatMessage] = Field(
        min_length=1,
        description="Conversation messages.",
    )
    stream: bool = Field(default=False, description="Whether to stream the response.")
    user: str | None = Field(default=None, description="Caller user identifier.")
    tools: list[OpenAIChatCompletionTool] | None = Field(
        default=None,
        description="Available tools.",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="Tool selection strategy.",
    )
    scheduled_route: Literal[
        "answer",
        "tool",
        "ragflow",
        "route",
        "mcp",
        "traffic",
        "service",
        "report",
    ] | None = Field(
        default=None,
        description="Optional execution route override for scheduled requests.",
    )
    brief_answer: bool = Field(
        default=True,
        description="Whether answer_node should use compact answer prompts, except reports.",
    )
    enable_thinking: bool | None = Field(
        default=None,
        description="Top-level thinking toggle for compatible Qwen-style models.",
    )
    chat_template_kwargs: OpenAIChatTemplateKwargs | None = Field(
        default=None,
        description="Optional nested chat_template_kwargs forwarded upstream.",
    )

    @property
    def resolved_enable_thinking(self) -> bool | None:
        """Resolve the effective thinking flag, preferring the top-level field."""

        if self.enable_thinking is not None:
            return self.enable_thinking
        if self.chat_template_kwargs is not None:
            return self.chat_template_kwargs.enable_thinking
        return None


class OpenAIChatCompletionAssistantMessage(BaseModel):
    """Assistant message in the response."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"] = Field(default="assistant", description="Assistant role.")
    content: str | None = Field(default=None, description="Assistant response content.")
    tool_calls: list[OpenAIChatCompletionToolCall] | None = Field(
        default=None,
        description="Tool calls returned by the assistant.",
    )


class OpenAIChatCompletionChoice(BaseModel):
    """Completion choice."""

    model_config = ConfigDict(extra="ignore")

    index: int = Field(default=0, description="Choice index.")
    message: OpenAIChatCompletionAssistantMessage = Field(description="Choice message.")
    finish_reason: str = Field(default="stop", description="Reason the completion stopped.")


class OpenAIChatCompletionUsage(BaseModel):
    """Token usage payload."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0, description="Prompt token count.")
    completion_tokens: int = Field(default=0, ge=0, description="Completion token count.")
    total_tokens: int = Field(default=0, ge=0, description="Total token count.")


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Response identifier.")
    object: Literal["chat.completion"] = Field(
        default="chat.completion",
        description="Response object type.",
    )
    created: int = Field(description="Unix timestamp when the response was created.")
    model: str = Field(description="Actual model name used.")
    choices: list[OpenAIChatCompletionChoice] = Field(description="Completion choices.")
    usage: OpenAIChatCompletionUsage = Field(description="Token usage.")
    system_fingerprint: str | None = Field(default=None, description="System fingerprint.")
