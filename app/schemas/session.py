"""会话领域数据模型。

负责描述会话创建、查询和列表返回的数据结构。
当前阶段不负责数据库表结构定义和权限校验。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SessionCreateRequest(BaseModel):
    """创建会话请求模型。"""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, description="会话标题。")
    user_id: str | None = Field(default=None, description="发起会话的用户标识。")


class SessionResponse(BaseModel):
    """会话响应模型。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="会话唯一标识。")
    title: str | None = Field(default=None, description="会话标题。")
    user_id: str | None = Field(default=None, description="所属用户标识。")
    status: str = Field(default="active", description="会话状态。")
    created_at: datetime = Field(description="创建时间。")
    updated_at: datetime = Field(description="更新时间。")


class SessionListResponse(BaseModel):
    """会话列表响应模型。"""

    model_config = ConfigDict(extra="forbid")

    items: list[SessionResponse] = Field(default_factory=list, description="会话列表。")
    total: int = Field(default=0, ge=0, description="会话总数。")
