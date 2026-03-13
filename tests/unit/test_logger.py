"""日志模块单元测试。"""

import logging

from app.core.logger import configure_logging, get_logger


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
