"""Checkpoint 存储模块。
负责把对话图的轻量快照写入 Redis，便于后续阶段扩展更细粒度的状态恢复。
当前阶段提供 Redis 优先、内存兜底的最小实现，不负责高可用和分布式一致性。
"""

from __future__ import annotations

from json import dumps, loads
from typing import Any, ClassVar

from redis.asyncio import Redis, from_url
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.logger import get_logger

LOGGER = get_logger(__name__)
CHECKPOINT_TTL_SECONDS = 3600


class CheckpointStore:
    """Redis checkpoint 存储器。"""

    _fallback_store: ClassVar[dict[str, dict[str, Any]]] = {}
    _unavailable_redis_urls: ClassVar[set[str]] = set()

    def __init__(self, redis_client: Redis | None = None) -> None:
        self._redis_client = redis_client
        self._is_redis_unavailable = False
        self._settings = get_settings()

    async def load(self, session_id: str) -> dict[str, Any] | None:
        """读取会话 checkpoint。"""

        redis_client = await self._get_redis_client()
        if redis_client is None:
            return self._fallback_store.get(session_id)

        try:
            raw_payload = await redis_client.get(self._build_key(session_id))
        except RedisError as exception:
            self._mark_redis_unavailable(exception)
            return self._fallback_store.get(session_id)

        if raw_payload is None:
            return None
        if isinstance(raw_payload, bytes):
            raw_payload = raw_payload.decode("utf-8")
        parsed_payload = loads(raw_payload)
        return parsed_payload if isinstance(parsed_payload, dict) else None

    async def save(
        self,
        *,
        session_id: str,
        payload: dict[str, Any],
        ttl_seconds: int = CHECKPOINT_TTL_SECONDS,
    ) -> None:
        """保存会话 checkpoint。"""

        self._fallback_store[session_id] = payload
        redis_client = await self._get_redis_client()
        if redis_client is None:
            return

        try:
            await redis_client.set(
                self._build_key(session_id),
                dumps(payload, ensure_ascii=False),
                ex=ttl_seconds,
            )
        except RedisError as exception:
            self._mark_redis_unavailable(exception)

    async def clear(self, session_id: str) -> None:
        """删除会话 checkpoint。"""

        self._fallback_store.pop(session_id, None)
        redis_client = await self._get_redis_client()
        if redis_client is None:
            return

        try:
            await redis_client.delete(self._build_key(session_id))
        except RedisError as exception:
            self._mark_redis_unavailable(exception)

    async def _get_redis_client(self) -> Redis | None:
        """延迟初始化 Redis 客户端，并在首次失败后禁用远程访问。"""

        if self._is_redis_unavailable or self._settings.redis_url in self._unavailable_redis_urls:
            self._is_redis_unavailable = True
            return None

        if self._redis_client is None:
            self._redis_client = from_url(
                self._settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )

        try:
            await self._redis_client.ping()
        except RedisError as exception:
            self._mark_redis_unavailable(exception)
            return None
        return self._redis_client

    def _mark_redis_unavailable(self, exception: Exception) -> None:
        """标记 Redis 当前不可用，并记录一次降级日志。"""

        if self._is_redis_unavailable:
            return
        self._is_redis_unavailable = True
        self._unavailable_redis_urls.add(self._settings.redis_url)
        LOGGER.warning("Redis checkpoint 不可用，已降级到进程内存。", exc_info=exception)

    @staticmethod
    def _build_key(session_id: str) -> str:
        """构造统一的 checkpoint key。"""

        return f"chat-agent:checkpoint:{session_id}"
