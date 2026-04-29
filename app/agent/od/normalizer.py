"""OD 文本清洗工具。

这里的清洗只处理“问题包装”和“端点边界噪声”，不负责判断地点是否真实存在。
这样可以兼容 /agent/driving 对未知 POI 的宽松解析能力。
"""

from __future__ import annotations

from app.agent.od.patterns import (
    ENDPOINT_PREFIX_PATTERN,
    ENDPOINT_SUFFIX_PATTERN,
    ROUTE_ACTION_SUFFIX_PATTERN,
    SPACE_PATTERN,
    WRAPPER_PREFIX_PATTERN,
)

# 只剥离端点首尾标点，避免影响地点内部字符。
_EDGE_PUNCTUATION = " ，,。.?？!！;；:：、"


def normalize_query(message: str) -> str:
    """标准化原始问题。

    只移除外层包装和空白，保留“从/到/往/去”等 OD 关系词，
    因为这些词是后续判断起终点边界的关键。
    """

    text = str(message or "").strip()
    while True:
        cleaned = WRAPPER_PREFIX_PATTERN.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned
    return SPACE_PATTERN.sub("", text)


def clean_endpoint(value: object | None) -> str:
    """清洗单个起点或终点片段。

    该函数不会查地点库，只负责把“我现在正在益农服务区”清成“益农服务区”，
    把“杭州去”清成“杭州”。
    """

    text = str(value or "").strip().strip(_EDGE_PUNCTUATION)
    if not text:
        return ""

    changed = True
    while changed:
        changed = False
        cleaned = ENDPOINT_PREFIX_PATTERN.sub("", text).strip().strip(_EDGE_PUNCTUATION)
        if cleaned != text and cleaned:
            # 循环剥离是为了处理“我现在正在从杭州”这类多层口语前缀。
            text = cleaned
            changed = True

    changed = True
    while changed:
        changed = False
        for pattern in (ENDPOINT_SUFFIX_PATTERN, ROUTE_ACTION_SUFFIX_PATTERN):
            cleaned = pattern.sub("", text).strip().strip(_EDGE_PUNCTUATION)
            if cleaned != text and cleaned:
                # destination 容易带上“去/多久/堵不堵”等尾巴，需要反复剥离。
                text = cleaned
                changed = True
                break

    return text.strip().strip(_EDGE_PUNCTUATION)
