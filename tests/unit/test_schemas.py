"""基础数据模型单元测试。"""

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.knowledge import KnowledgeSearchRequest
from app.schemas.session import SessionListResponse


def test_chat_request_uses_independent_metadata_defaults() -> None:
    """验证请求模型的默认元数据不会在实例间共享。"""

    first_request = ChatRequest(user_message="你好")
    second_request = ChatRequest(user_message="世界")

    first_request.metadata["source"] = "unit-test"

    assert second_request.metadata == {}


def test_knowledge_search_request_validates_top_k_range() -> None:
    """验证知识检索请求会校验 top_k 范围。"""

    with pytest.raises(ValidationError):
        KnowledgeSearchRequest(query="测试问题", top_k=0)


def test_session_list_response_defaults_to_empty_collection() -> None:
    """验证会话列表响应的默认值符合预期。"""

    response = SessionListResponse()

    assert response.items == []
    assert response.total == 0
