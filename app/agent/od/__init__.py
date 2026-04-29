"""OD 起终点解析模块对外入口。

本包负责把用户口语化路线问题解析成干净的 origin/destination，
并在调用路线接口前提供参数校验能力。
"""

from app.agent.od.guard import is_valid_endpoint, validate_route_arguments
from app.agent.od.models import OdResolution
from app.agent.od.resolver import OdResolver, resolve_od

__all__ = [
    "OdResolution",
    "OdResolver",
    "is_valid_endpoint",
    "resolve_od",
    "validate_route_arguments",
]
