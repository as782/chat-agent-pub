"""OpenAI 兼容接口数据模型。
负责定义 OpenAI Chat Completions 兼容请求与响应结构。
当前阶段只覆盖文本消息、工具调用和流式 chunk，不负责 Responses API 与音频输出。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAITextContentPart(BaseModel):
    """OpenAI 文本内容分片模型。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["text"] = Field(default="text", description="内容分片类型。")
    text: str = Field(description="文本内容。")


class OpenAIChatCompletionToolFunction(BaseModel):
    """OpenAI 兼容函数工具定义。"""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="工具名称。")
    description: str | None = Field(default=None, description="工具说明。")
    parameters: dict[str, Any] = Field(default_factory=dict, description="工具参数 JSON Schema。")


class OpenAIChatCompletionTool(BaseModel):
    """OpenAI 兼容工具定义。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = Field(default="function", description="工具类型。")
    function: OpenAIChatCompletionToolFunction = Field(description="函数工具定义。")


class OpenAIChatCompletionToolCallFunction(BaseModel):
    """OpenAI 兼容工具调用中的函数体。"""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="函数名称。")
    arguments: str = Field(description="JSON 字符串格式的函数参数。")


class OpenAIChatCompletionToolCall(BaseModel):
    """OpenAI 兼容工具调用模型。"""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="工具调用标识。")
    type: Literal["function"] = Field(default="function", description="工具调用类型。")
    function: OpenAIChatCompletionToolCallFunction = Field(description="函数调用负载。")


class OpenAIChatMessage(BaseModel):
    """OpenAI 兼容消息模型。"""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"] = Field(
        description="消息角色。"
    )
    content: str | list[OpenAITextContentPart] | None = Field(
        default=None,
        description="消息内容，当前仅处理纯文本。",
    )
    name: str | None = Field(default=None, description="消息发送方名称。")
    tool_call_id: str | None = Field(default=None, description="tool 消息对应的工具调用标识。")
    tool_calls: list[OpenAIChatCompletionToolCall] | None = Field(
        default=None,
        description="assistant 消息携带的工具调用列表。",
    )


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions 兼容请求模型。"""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(description="请求使用的模型名称。")
    messages: list[OpenAIChatMessage] = Field(min_length=1, description="对话消息列表。")
    stream: bool = Field(default=False, description="是否使用流式输出。")
    user: str | None = Field(default=None, description="调用方用户标识。")
    tools: list[OpenAIChatCompletionTool] | None = Field(
        default=None,
        description="允许模型使用的工具列表。",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="工具选择策略，可为 auto、none、required 或指定函数。",
    )


class OpenAIChatCompletionAssistantMessage(BaseModel):
    """OpenAI 兼容助手消息模型。"""

    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"] = Field(default="assistant", description="助手角色。")
    content: str | None = Field(default=None, description="助手回答内容。")
    tool_calls: list[OpenAIChatCompletionToolCall] | None = Field(
        default=None,
        description="助手返回的工具调用列表。",
    )


class OpenAIChatCompletionChoice(BaseModel):
    """OpenAI 兼容候选回答模型。"""

    model_config = ConfigDict(extra="ignore")

    index: int = Field(default=0, description="候选回答索引。")
    message: OpenAIChatCompletionAssistantMessage = Field(description="候选回答消息。")
    finish_reason: str = Field(default="stop", description="完成原因。")


class OpenAIChatCompletionUsage(BaseModel):
    """OpenAI 兼容 token 使用量模型。"""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0, description="提示词 token 数。")
    completion_tokens: int = Field(default=0, ge=0, description="生成 token 数。")
    total_tokens: int = Field(default=0, ge=0, description="总 token 数。")


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI Chat Completions 兼容响应模型。"""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="响应唯一标识。")
    object: Literal["chat.completion"] = Field(
        default="chat.completion",
        description="响应对象类型。",
    )
    created: int = Field(description="响应创建时间戳。")
    model: str = Field(description="实际使用的模型名称。")
    choices: list[OpenAIChatCompletionChoice] = Field(description="候选回答列表。")
    usage: OpenAIChatCompletionUsage = Field(description="token 使用量。")
    system_fingerprint: str | None = Field(default=None, description="系统指纹。")
