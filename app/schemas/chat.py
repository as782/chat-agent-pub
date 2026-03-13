"""对话领域数据模型。

负责定义对话请求和响应的数据结构，供 API、Service 和 Agent 层复用。
当前阶段不负责数据库实体映射和流式响应协议。
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """对话请求模型。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, description="会话标识。")
    user_message: str = Field(min_length=1, description="用户输入的消息内容。")
    stream: bool = Field(default=False, description="是否启用流式响应。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="请求扩展元数据。")


class ChatResponse(BaseModel):
    """对话响应模型。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="当前对话所属会话标识。")
    answer: str = Field(description="系统生成的回答内容。")
    used_knowledge: bool = Field(default=False, description="是否命中了知识库能力。")
    used_tools: list[str] = Field(default_factory=list, description="本次调用使用到的工具列表。")
