"""日志模块。

负责统一应用日志输出格式，并提供轻量的日志对象获取方法。
当前阶段不负责日志落盘、链路追踪和外部日志平台上报。
"""

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def resolve_log_level(is_debug: bool) -> int:
    """根据调试开关推导日志级别。"""

    return logging.DEBUG if is_debug else logging.INFO


def configure_logging(is_debug: bool) -> None:
    """配置全局日志格式与日志级别。"""

    logging.basicConfig(
        level=resolve_log_level(is_debug=is_debug),
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )


def get_logger(logger_name: str) -> logging.Logger:
    """获取指定名称的日志对象。"""

    return logging.getLogger(logger_name)
