import re


_TRAFFIC_EVENT_WORDS = (
    "事故",
    "追尾",
    "剐蹭",
    "施工",
    "修路",
    "封闭",
    "封路",
    "管制",
    "限行",
    "分流",
    "绕行",
    "塌方",
    "积水",
    "结冰",
    "大雾",
)

_TRAFFIC_STATUS_WORDS = (
    "堵车",
    "堵不堵",
    "堵吗",
    "拥堵",
    "缓行",
    "压车",
    "车多",
    "车流量大",
    "排队",
    "排队长吗",
    "通行",
    "畅通",
    "是否畅通",
    "是否拥堵",
    "正常通行",
    "正常吗",
    "好走吗",
    "好不好走",
    "顺不顺",
    "恢复了吗",
    "恢复通行",
    "什么时候恢复",
    "多久恢复",
    "路况",
)

_TRAFFIC_SCENE_WORDS = (
    "高速",
    "高架",
    "站口",
    "入口",
    "出口",
    "枢纽",
    "互通",
    "立交",
    "匝道",
    "主线",
    "辅路",
    "路段",
    "这一段",
    "那一段",
    "前方",
    "后方",
)

_TRAFFIC_QUERY_WORDS = (
    "能走吗",
    "能走不",
    "可以走吗",
    "可不可以走",
    "能不能走",
    "通不通",
    "通了吗",
    "堵不堵",
    "堵吗",
    "怎么样",
    "咋样",
    "什么情况",
    "啥情况",
    "现在怎么样",
    "现在咋样",
    "看一下",
    "查一下",
    "看看",
    "能上吗",
    "可以上吗",
    "能不能上",
    "能下吗",
    "可以下吗",
    "能不能下",
    "上得去吗",
    "下得来吗",
    "封没封",
    "开着吗",
)

# 一些容易误判成“路况”的非交通场景
_NEGATIVE_HINTS = (
    "天气怎么样",
    "股票怎么样",
    "基金怎么样",
    "方案怎么样",
    "代码怎么样",
    "这个人怎么样",
    "这个车怎么样",
    "订单",
    "物流",
    "快递",
    "外卖",
    "酒店",
    "房间",
    "面试",
    "简历",
    "论文",
    "作业",
)

# 明显 OD / 通行状态问法
_OD_TRAFFIC_PATTERNS = [
    r".+到.+(能走吗|能不能走|可以走吗|通不通|通了吗|堵不堵|堵吗|怎么样|咋样|好走吗|多久|需要多久)",
    r".+去.+(能走吗|能不能走|可以走吗|通不通|通了吗|堵不堵|堵吗|怎么样|咋样|好走吗|多久|需要多久)",
    r".+往.+方向(堵不堵|堵吗|怎么样|咋样|能走吗|通不通|好走吗)",
    r".+(收费站|站口|入口|出口).+(能上吗|能下吗|开着吗|封没封|通了吗|封了吗)",
    r".+(高速|高架|快速路|国道|省道|主线|匝道|路段).+(堵不堵|堵吗|通不通|怎么样|咋样|能走吗)",
]

# 类似 “郑州到开封” 但没明显交通词时，需要和这些状态词搭配
_OD_STATUS_HINTS = (
    "能走吗",
    "能不能走",
    "可以走吗",
    "通不通",
    "通了吗",
    "堵不堵",
    "堵吗",
    "好走吗",
    "多久",
    "需要多久",
    "几小时",
    "几个小时",
    "多久能到",
)


def _normalize_message(message: str) -> str:
    if not message:
        return ""
    msg = str(message).strip().lower()
    msg = re.sub(r"\s+", "", msg)
    return msg


def _contains_any(msg: str, words: tuple[str, ...]) -> bool:
    return any(w in msg for w in words)


def _looks_like_pure_negative_case(msg: str) -> bool:
    # 完全没有交通场景时，拦一下明显非交通问法
    if _contains_any(msg, _NEGATIVE_HINTS):
        if not _contains_any(msg, _TRAFFIC_SCENE_WORDS + _TRAFFIC_EVENT_WORDS + _TRAFFIC_STATUS_WORDS):
            return True
    return False


def _looks_like_od_traffic(msg: str) -> bool:
    # 正则兜底
    if any(re.search(pattern, msg) for pattern in _OD_TRAFFIC_PATTERNS):
        return True

    # 更宽松一点：出现“到/去/往...方向” + 状态词，也算 traffic
    has_od_shape = (
        ("到" in msg and len(msg.split("到", 1)[0]) > 0 and len(msg.split("到", 1)[1]) > 0)
        or ("去" in msg and len(msg.split("去", 1)[0]) > 0 and len(msg.split("去", 1)[1]) > 0)
        or ("往" in msg and "方向" in msg)
    )
    if has_od_shape and _contains_any(msg, _OD_STATUS_HINTS + _TRAFFIC_QUERY_WORDS):
        return True

    return False


def looks_like_traffic_query_v1(message: str) -> bool:
    msg = _normalize_message(message)
    if not msg:
        return False

    # 1. 明显非交通误判拦截
    if _looks_like_pure_negative_case(msg):
        return False

    # 2. 明确事故 / 施工 / 封路 等事件词
    if _contains_any(msg, _TRAFFIC_EVENT_WORDS):
        return True

    # 3. 明确路况状态词
    if _contains_any(msg, _TRAFFIC_STATUS_WORDS):
        return True

    # 4. 高速场景 + 查询词
    if _contains_any(msg, _TRAFFIC_SCENE_WORDS) and _contains_any(msg, _TRAFFIC_QUERY_WORDS):
        return True

    # 5. 典型 OD + 路况问法
    if _looks_like_od_traffic(msg):
        return True

    return False

if __name__ == "__main__":
    tests = [
        "郑州到开封能走吗",
        "济广高速往六安方向堵不堵",
        "合肥南收费站现在能上吗",
        "金寨出口能下吗",
        "前方什么情况",
        "是不是出事故了",
        "现在路况咋样",
        "南京到苏州多久",
        "服务区挤不挤",
        "今天天气怎么样",
        "帮我看一下订单",
        "这个方案怎么样",
        "看看杭州到金华"
        "杭州到金华"
    ]

    for t in tests:
        print(t, looks_like_traffic_query_v1(t))