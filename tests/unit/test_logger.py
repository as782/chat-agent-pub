"""日志模块单元测试。"""

from __future__ import annotations

import logging

from app.core.logger import (
    MAX_MCP_SSE_VERBOSE_MESSAGE_LENGTH,
    MCP_SSE_TRUNCATED_MESSAGE_SUFFIX,
    MCP_SSE_VERBOSE_MESSAGE_PREFIX,
    McpSseVerboseMessageFilter,
    configure_logging,
    get_logger,
)


def test_configure_logging_enables_debug_level_in_debug_mode() -> None:
    """验证调试环境会启用调试级日志。"""

    configure_logging(is_debug=True)
    logger = get_logger("tests.logger")

    assert logger.getEffectiveLevel() == logging.DEBUG


def test_configure_logging_uses_info_level_in_non_debug_mode() -> None:
    """验证非调试环境默认使用信息级日志。"""

    configure_logging(is_debug=False)
    logger = get_logger("tests.logger")

    assert logger.getEffectiveLevel() == logging.INFO


def test_configure_logging_reduces_noisy_third_party_loggers() -> None:
    """验证高噪声第三方日志会被收敛到告警级别。"""

    configure_logging(is_debug=True)

    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("sqlalchemy.engine.Engine").getEffectiveLevel() == logging.WARNING


def test_mcp_sse_verbose_message_filter_truncates_large_server_messages() -> None:
    """验证过长的 MCP SSE 原始报文会被截断。"""

    verbose_message = (
        MCP_SSE_VERBOSE_MESSAGE_PREFIX + " " + ("x" * (MAX_MCP_SSE_VERBOSE_MESSAGE_LENGTH + 50))
    )
    record = logging.LogRecord(
        name="mcp.client.sse",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg=verbose_message,
        args=(),
        exc_info=None,
    )

    filter_instance = McpSseVerboseMessageFilter()

    assert filter_instance.filter(record) is True
    assert record.msg.endswith(MCP_SSE_TRUNCATED_MESSAGE_SUFFIX)
    assert len(record.msg) > MAX_MCP_SSE_VERBOSE_MESSAGE_LENGTH
    assert len(record.msg) < len(verbose_message)
