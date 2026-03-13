"""消息领域数据模型。

负责定义消息历史查询相关的数据结构，供 API 和 Service 层复用。
当前阶段不负责流式事件协议和消息编辑能力。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageResponse(BaseModel):
    """消息响应模型。"""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(description="消息唯一标识。")
    session_id: str = Field(description="所属会话标识。")
    role: str = Field(description="消息角色，例如 user 或 assistant。")
    content: str = Field(description="消息内容。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="消息扩展元数据。")
    created_at: datetime = Field(description="消息创建时间。")


class MessageListResponse(BaseModel):
    """消息列表响应模型。"""

    model_config = ConfigDict(extra="forbid")

    items: list[MessageResponse] = Field(default_factory=list, description="消息列表。")
    total: int = Field(default=0, ge=0, description="消息总数。")
