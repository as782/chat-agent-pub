"""RAGFlow 对话客户端模块。

负责封装 RAGFlow chat assistant completion 接口，供知识库透传调用场景复用。
当前阶段仅做最小非流式透传，不负责协议转换增强。
"""

from __future__ import annotations

from typing import Any

from app.knowledge.ragflow.client import RagflowClient


class RagflowChatClient:
    """RAGFlow 对话客户端。"""

    def __init__(self, ragflow_client: RagflowClient | None = None) -> None:
        self._ragflow_client = ragflow_client or RagflowClient()

    async def create_chat_completion(
        self,
        *,
        chat_id: str,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """向指定 chat assistant 发起 completion 请求。"""

        response_data = await self._ragflow_client.request(
            "POST",
            f"/api/v1/chats/{chat_id}/completions",
            json_body=request_payload,
            expect_envelope=False,
        )
        return response_data if isinstance(response_data, dict) else {}
