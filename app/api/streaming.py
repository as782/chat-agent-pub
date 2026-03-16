"""流式响应辅助模块。
负责在返回 StreamingResponse 前预取首个数据块，避免首包异常导致连接直接中断。
当前阶段不负责复杂背压控制和多路流复用。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import status

from app.core.exceptions import AppException


async def prime_stream_iterator(stream_iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    """预取首个流式块，确保响应开始前就暴露首包错误。"""

    try:
        first_chunk = await anext(stream_iterator)
    except StopAsyncIteration as exception:
        raise AppException(
            "流式响应未返回任何数据。",
            error_code="invalid_llm_response",
            status_code=status.HTTP_502_BAD_GATEWAY,
        ) from exception

    async def wrapped_iterator() -> AsyncIterator[str]:
        """先返回已预取的首块，再继续透传后续块。"""

        yield first_chunk
        async for chunk in stream_iterator:
            yield chunk

    return wrapped_iterator()
