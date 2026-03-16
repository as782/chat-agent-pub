"""OpenAI 兼容接口数据模型。

负责定义 OpenAI Chat Completions 兼容请求和响应结构。
当前阶段仅覆盖文本对话的最小兼容子集，不负责工具调用和流式事件协议。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAITextContentPart(BaseModel):
    """OpenAI 文本内容分片模型。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["text"] = Field(default="text", description="内容分片类型。")
    text: str = Field(description="文本内容。")


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


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions 兼容请求模型。"""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(description="请求使用的模型名称。")
    messages: list[OpenAIChatMessage] = Field(min_length=1, description="对话消息列表。")
    stream: bool = Field(default=False, description="是否使用流式输出。")
    user: str | None = Field(default=None, description="调用方用户标识。")


class OpenAIChatCompletionAssistantMessage(BaseModel):
    """OpenAI 兼容助手消息模型。"""

    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"] = Field(default="assistant", description="助手角色。")
    content: str = Field(description="助手回答内容。")


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
