"""OD 起终点解析结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PlaceCandidate:
    """文本中命中的地点候选。

    候选可能来自 Aho-Corasick 精确词典，也可能来自“X收费站/X服务区”形态兜底。
    resolver 会基于候选位置和连接词上下文选择最终起终点。
    """

    text: str
    canonical_name: str
    place_type: str
    start: int
    end: int
    match_type: str
    priority: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OdResolution:
    """单次 OD 解析结果。

    origin/destination 是最终可用于路线接口的起终点；raw_* 保留规则命中的原始片段，
    match_type 用于排查结果来自设施目录还是结构化文本兜底。
    """

    # 清洗、归一化后的起点和终点。
    origin: str | None
    destination: str | None
    # 当前解析结果的置信度，仅用于内部排序和日志观察，不直接作为业务答案。
    confidence: float
    # 解析来源，便于和 planner metadata、旧正则兜底区分。
    source: str
    # 从原句中截取到的原始起终点片段。
    raw_origin: str | None = None
    raw_destination: str | None = None
    # 起终点匹配类型，例如 facility_service_area、facility_toll_station、structured_text。
    origin_match_type: str | None = None
    destination_match_type: str | None = None
    # 非阻断型告警，用于日志和后续排查。
    warnings: list[str] = field(default_factory=list)
    # 本次解析看到的地点候选，用于日志和问题排查。
    candidates: list[PlaceCandidate] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """是否已经同时解析出起点和终点。"""

        return bool(self.origin and self.destination)
