"""日志模块。
负责统一应用日志输出格式，并对高噪声第三方日志做分级收敛。
当前阶段不负责日志落盘、链路追踪和外部日志平台上报。
"""

from __future__ import annotations

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
MCP_SSE_VERBOSE_MESSAGE_PREFIX = "Received server message:"
MCP_SSE_TRUNCATED_MESSAGE_SUFFIX = " ...<truncated>"
MAX_MCP_SSE_VERBOSE_MESSAGE_LENGTH = 240
NOISY_LOGGER_LEVELS = {
    # 这些库在本地调试时会输出大量底层网络和数据库细节，默认收敛到告警级别。
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "openai": logging.WARNING,
    "openai._base_client": logging.WARNING,
    "sqlalchemy.engine": logging.WARNING,
    "sqlalchemy.pool": logging.WARNING,
    "aiosqlite": logging.WARNING,
}


class McpSseVerboseMessageFilter(logging.Filter):
    """精简 MCP SSE 客户端过长的原始报文日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """截断过长的 SSE 原始报文，避免控制台被完整 payload 淹没。"""

        if record.name != "mcp.client.sse":
            return True

        rendered_message = record.getMessage()
        if not rendered_message.startswith(MCP_SSE_VERBOSE_MESSAGE_PREFIX):
            return True

        if len(rendered_message) <= MAX_MCP_SSE_VERBOSE_MESSAGE_LENGTH:
            return True

        record.msg = (
            rendered_message[:MAX_MCP_SSE_VERBOSE_MESSAGE_LENGTH] + MCP_SSE_TRUNCATED_MESSAGE_SUFFIX
        )
        record.args = ()
        return True


def resolve_log_level(is_debug: bool) -> int:
    """根据调试开关推导应用默认日志级别。"""

    return logging.DEBUG if is_debug else logging.INFO


def configure_logging(is_debug: bool) -> None:
    """配置全局日志格式、日志级别和第三方噪声日志控制。"""

    logging.basicConfig(
        level=resolve_log_level(is_debug=is_debug),
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )

    for logger_name, logger_level in NOISY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    sse_logger = logging.getLogger("mcp.client.sse")
    sse_logger.filters = [
        logger_filter
        for logger_filter in sse_logger.filters
        if not isinstance(logger_filter, McpSseVerboseMessageFilter)
    ]
    sse_logger.addFilter(McpSseVerboseMessageFilter())


def get_logger(logger_name: str) -> logging.Logger:
    """获取指定名称的日志对象。"""

    return logging.getLogger(logger_name)
