"""知识库领域数据模型。

负责定义知识库检索、数据集管理和 RAGFlow 透传所需的数据结构。
当前阶段不负责向量检索实现和文档解析流程。
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.openai_compat import OpenAIChatCompletionRequest


class KnowledgeSearchRequest(BaseModel):
    """知识检索请求模型。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="用户检索问题。")
    dataset_ids: list[str] = Field(default_factory=list, description="目标数据集标识列表。")
    top_k: int = Field(default=5, ge=1, le=20, description="返回的候选条数。")


class KnowledgeSearchResult(BaseModel):
    """知识检索结果模型。"""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="文档标识。")
    chunk_id: str = Field(description="分片标识。")
    score: float = Field(description="召回得分。")
    content: str = Field(description="召回内容。")
    source: str | None = Field(default=None, description="内容来源。")


class KnowledgeSearchResponse(BaseModel):
    """知识检索响应模型。"""

    model_config = ConfigDict(extra="forbid")

    results: list[KnowledgeSearchResult] = Field(default_factory=list, description="检索结果列表。")


class KnowledgeDatasetItem(BaseModel):
    """知识库数据集模型。"""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(description="数据集标识。")
    dataset_name: str = Field(description="数据集名称。")
    is_enabled: bool = Field(default=True, description="当前是否启用。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="原始数据集元数据。")


class KnowledgeDatasetListResponse(BaseModel):
    """知识库数据集列表响应。"""

    model_config = ConfigDict(extra="forbid")

    items: list[KnowledgeDatasetItem] = Field(default_factory=list, description="数据集列表。")
    synced_count: int = Field(default=0, ge=0, description="本次同步更新的数据集数量。")


class KnowledgeDocumentItem(BaseModel):
    """知识库文档模型。"""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="文档标识。")
    document_name: str = Field(description="文档名称。")
    status: str = Field(description="文档状态。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="文档原始元数据。")


class KnowledgeDocumentListResponse(BaseModel):
    """知识库文档列表响应。"""

    model_config = ConfigDict(extra="forbid")

    items: list[KnowledgeDocumentItem] = Field(default_factory=list, description="文档列表。")


class KnowledgeChatRequest(OpenAIChatCompletionRequest):
    """知识库聊天透传请求模型。"""

    model_config = ConfigDict(extra="forbid")

    chat_id: str = Field(min_length=1, description="RAGFlow chat assistant 标识。")
