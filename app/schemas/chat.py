"""对话领域数据模型。
负责定义内部聊天请求结构，供 API、Service 和 Agent 层复用。
当前阶段不负责 OpenAI 兼容响应协议定义，该部分由独立 schema 模块承接。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
