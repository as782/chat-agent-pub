"""日志模块。

负责统一应用日志输出格式，并支持控制台输出、文件持久化和按时间轮转。
当前阶段不负责外部日志平台上报。
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
MCP_SSE_VERBOSE_MESSAGE_PREFIX = "Received server message:"
MCP_SSE_TRUNCATED_MESSAGE_SUFFIX = " ...<truncated>"
MAX_MCP_SSE_VERBOSE_MESSAGE_LENGTH = 240
LOG_FILE_SUFFIX_BY_ROTATION = {
    "S": "%Y-%m-%d_%H-%M-%S",
    "M": "%Y-%m-%d_%H-%M",
    "H": "%Y-%m-%d_%H",
    "D": "%Y-%m-%d",
    "midnight": "%Y-%m-%d",
}
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

if TYPE_CHECKING:
    from app.core.config import Settings


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


def _reset_logger_handlers(logger: logging.Logger) -> None:
    """清理 logger 已有处理器，避免重复打印和文件句柄泄漏。"""

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _build_formatter() -> logging.Formatter:
    """构建统一的日志格式化器。"""

    return logging.Formatter(DEFAULT_LOG_FORMAT)


def _create_console_handler(formatter: logging.Formatter) -> logging.Handler:
    """创建控制台日志处理器。"""

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    return console_handler


def _resolve_log_file_suffix(rotate_when: str) -> str:
    """根据轮转策略生成日志文件后缀格式。"""

    return LOG_FILE_SUFFIX_BY_ROTATION.get(rotate_when, "%Y-%m-%d")


def _create_file_handler(settings: Settings, formatter: logging.Formatter) -> logging.Handler:
    """创建按时间轮转的文件日志处理器。"""

    log_dir = Path(settings.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / settings.log_file_name

    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when=settings.log_rotate_when,
        interval=settings.log_rotate_interval,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.suffix = _resolve_log_file_suffix(settings.log_rotate_when)
    file_handler.setFormatter(formatter)
    return file_handler


def _configure_uvicorn_loggers() -> None:
    """让 Uvicorn 日志统一复用根日志处理器。"""

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        _reset_logger_handlers(uvicorn_logger)
        uvicorn_logger.propagate = True


def configure_logging(settings: Settings) -> None:
    """配置全局日志格式、控制台输出、文件持久化与时间轮转策略。"""

    root_logger = logging.getLogger()
    _reset_logger_handlers(root_logger)
    root_logger.setLevel(resolve_log_level(is_debug=settings.is_debug))

    formatter = _build_formatter()
    root_logger.addHandler(_create_console_handler(formatter))
    if settings.enable_file_logging:
        root_logger.addHandler(_create_file_handler(settings, formatter))

    for logger_name, logger_level in NOISY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    _configure_uvicorn_loggers()

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
