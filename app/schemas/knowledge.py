"""知识库领域数据模型。

负责定义知识库检索请求与返回结构，供后续 RAGFlow 接入层复用。
当前阶段不负责向量检索实现和文档解析流程。
"""

from pydantic import BaseModel, ConfigDict, Field


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
