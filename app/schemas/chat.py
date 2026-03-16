"""对话领域数据模型。
负责定义对话请求、响应和流式事件的数据结构，供 API、Service 和 Agent 层复用。
当前阶段不负责数据库实体映射和完整 OpenAI 协议定义。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatToolCallResponse(BaseModel):
    """内部聊天接口的工具调用结果。"""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="工具调用唯一标识。")
    tool_name: str = Field(description="本次调用的工具名称。")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具调用参数。")
    output: str = Field(description="工具执行后的输出。")


class ChatRequest(BaseModel):
    """对话请求模型。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, description="会话标识。")
    user_message: str = Field(min_length=1, description="用户输入的消息内容。")
    stream: bool = Field(default=False, description="是否启用流式响应。")
    model: str | None = Field(default=None, description="本次请求指定的模型名称。")
    enable_tools: bool = Field(default=False, description="是否允许模型调用内置工具。")
    tool_names: list[str] = Field(
        default_factory=list,
        description="允许使用的内置工具名称列表，留空时表示启用全部内置工具。",
    )
    tool_choice: str | None = Field(
        default=None,
        description="工具选择策略，可选 auto、none、required 或具体工具名。",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="请求扩展元数据。")


class ChatResponse(BaseModel):
    """对话响应模型。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="当前对话所属会话标识。")
    answer: str = Field(description="系统生成的回答内容。")
    model: str = Field(description="实际使用的模型名称。")
    finish_reason: str = Field(description="本轮回答结束原因。")
    used_knowledge: bool = Field(default=False, description="是否命中了知识库能力。")
    used_tools: list[str] = Field(default_factory=list, description="本次调用使用到的工具列表。")
    tool_calls: list[ChatToolCallResponse] = Field(
        default_factory=list,
        description="本次请求实际执行的工具调用明细。",
    )
