import random
from collections import Counter

from app.agent.tool_traffic_intent import looks_like_traffic_query_v1


def build_traffic_testset_v1_aligned(seed: int = 42, total: int = 1000):
    random.seed(seed)

    # -----------------------------
    # 基础词表
    # -----------------------------
    cities = [
        "郑州", "开封", "洛阳", "南京", "苏州", "无锡", "常州", "合肥", "六安", "芜湖",
        "武汉", "黄石", "孝感", "杭州", "绍兴", "宁波", "广州", "佛山", "东莞", "深圳",
        "成都", "德阳", "绵阳", "西安", "咸阳", "宝鸡", "长沙", "株洲", "湘潭", "昆明",
        "曲靖", "大理", "南宁", "柳州", "桂林", "南昌", "九江", "赣州", "福州", "泉州",
    ]

    highways = [
        "京港澳高速", "连霍高速", "沪蓉高速", "沪昆高速", "济广高速", "宁洛高速",
        "大广高速", "福银高速", "包茂高速", "二广高速", "长深高速", "沈海高速",
        "杭瑞高速", "京沪高速", "青兰高速", "兰海高速"
    ]

    toll_stations = [
        "合肥南收费站", "六安北收费站", "郑州东收费站", "苏州新区收费站", "南京南收费站",
        "武汉西收费站", "长沙北收费站", "佛山收费站", "成都东收费站", "西安北收费站",
        "开封收费站", "株洲西收费站", "柳州东收费站", "泉州南收费站"
    ]

    entrances = [
        "合肥南入口", "六安北入口", "郑州东入口", "苏州新区入口", "南京南入口",
        "武汉西入口", "长沙北入口", "佛山入口", "成都东入口", "西安北入口"
    ]

    exits = [
        "金寨出口", "庐江出口", "开封出口", "苏州出口", "常州出口",
        "黄石出口", "株洲出口", "佛山出口", "德阳出口", "咸阳出口"
    ]

    # v1 当前更稳妥支持的路段类
    road_sections = [
        "前方路段", "后方路段", "主线", "匝道", "互通", "枢纽", "高架", "路段"
    ]

    directions = [
        "往六安方向", "往郑州方向", "往南京方向", "往苏州方向", "往武汉方向",
        "往黄石方向", "往绍兴方向", "往佛山方向", "往德阳方向", "往咸阳方向"
    ]

    traffic_status = [
        "能走吗", "能不能走", "可以走吗", "通不通", "通了吗", "堵不堵", "堵吗",
        "怎么样", "咋样", "好走吗", "什么情况", "啥情况", "多久", "多久能到", "需要多久"
    ]

    traffic_events = [
        "是不是出事故了", "是不是追尾了", "是不是施工了", "是不是封路了",
        "有没有管制", "有没有分流", "是不是缓行", "是不是压车", "是不是排队了",
        "有没有封闭", "有没有限行", "恢复了吗", "什么时候恢复"
    ]

    # 注意：这里去掉了当前 v1 不稳定/不支持的正样本
    traffic_direct = [
        "现在路况咋样",
        "现在路况怎么样",
        "现在堵不堵",
        "现在通行正常吗",
        "现在能正常通行吗",
        "前方什么情况",
        "后方什么情况",
        "主线堵不堵",
        "匝道堵不堵",
        "入口开着吗",
        "出口能下吗",
    ]

    spoken_prefix = [
        "", "请问", "麻烦问下", "帮我看下", "帮忙看看", "老师", "主播", "你好",
        "咨询一下", "想问下"
    ]

    spoken_suffix = [
        "", "啊", "呀", "吗", "不", "呢", "哈", "谢谢", "麻烦了"
    ]

    punctuation = ["", "？", "?", "。。。", "，", ""]

    negative_weather = [
        "今天天气怎么样", "明天会下雨吗", "气温多少度", "这两天有雾吗",
        "今天冷不冷", "明天热不热", "要不要带伞", "空气质量怎么样"
    ]

    negative_order = [
        "帮我看一下订单", "这个订单到哪了", "快递到哪了", "物流更新了吗",
        "外卖什么时候到", "我的包裹派送了吗", "退款什么时候到账", "订单怎么还没发货"
    ]

    negative_review = [
        "这个方案怎么样", "这版代码怎么样", "这个产品怎么样", "这个人怎么样",
        "这份简历怎么样", "论文写得怎么样", "这个酒店怎么样", "这个房间怎么样"
    ]

    negative_misc = [
        "现在股市怎么样", "基金还能买吗", "今天大盘咋样", "面试情况怎么样",
        "今天心情怎么样", "这个电影好看吗", "今晚吃什么", "帮我看看作业"
    ]

    # 这些是“以前算正样本，但你当前 v1 不覆盖”的句子
    # 现在统一放到 hard_negative，避免污染召回
    unsupported_semantic_traffic = [
        "服务区挤不挤",
        "服务区现在能不能进",
        "东莞服务站里面能进吗",
        "巢湖服务区里面还有位置吗",
        "省道咋样",
        "国道通不通",
        "下雪还能走吗",
        "收费站堵不堵",
    ]

    # -----------------------------
    # 包装函数
    # -----------------------------
    def wrap(text: str) -> str:
        pre = random.choice(spoken_prefix)
        suf = random.choice(spoken_suffix)
        punc = random.choice(punctuation)
        spaces = ["", " ", "  "]

        if random.random() < 0.15:
            text = text.replace("现在", random.choice(["现在", "这会儿", "这时候"]))
        if random.random() < 0.12:
            text = text.replace("怎么样", random.choice(["怎么样", "咋样"]))
        if random.random() < 0.12:
            text = text.replace("能不能", random.choice(["能不能", "可不可以"]))
        if random.random() < 0.08:
            text = text.replace("收费站", random.choice(["收费站", "站口"]))

        join1 = random.choice(spaces)
        join2 = random.choice(spaces)
        return f"{pre}{join1}{text}{join2}{suf}{punc}".strip()

    def fill_cases(target_count: int, builder, label: bool, tag: str):
        """保证补足数量，不再出现 total 不够。"""
        cases = set()
        tries = 0
        max_tries = target_count * 50

        while len(cases) < target_count and tries < max_tries:
            text = builder()
            cases.add((text, label, tag))
            tries += 1

        return list(cases)

    # -----------------------------
    # 正样本生成（只保留 v1 已覆盖范围）
    # -----------------------------
    true_target = total // 2
    false_target = total - true_target

    true_cases = []

    # 1. OD 问法
    od_templates = [
        "{a}到{b}{q}",
        "{a}去{b}{q}",
        "从{a}到{b}{q}",
        "{a}往{b}方向{q}",
        "从{a}往{b}方向{q}",
        "{a}到{b}现在{q}",
        "{a}去{b}现在{q}",
        "{a}到{b}路上{q}",
    ]

    def build_od_case():
        a, b = random.sample(cities, 2)
        tpl = random.choice(od_templates)
        q = random.choice(traffic_status)
        return wrap(tpl.format(a=a, b=b, q=q))

    true_cases += fill_cases(180, build_od_case, True, "od")

    # 2. 高速 + 方向
    hw_templates = [
        "{hw}{d}{q}",
        "{hw}现在{q}",
        "{hw}上面{q}",
        "{hw}今天{q}",
        "{hw}{d}现在{q}",
        "{hw}{d}路况{q}",
    ]

    def build_highway_case():
        hw = random.choice(highways)
        d = random.choice(directions)
        q = random.choice(["堵不堵", "堵吗", "怎么样", "咋样", "通不通", "能走吗", "好走吗", "什么情况"])
        return wrap(random.choice(hw_templates).format(hw=hw, d=d, q=q))

    true_cases += fill_cases(120, build_highway_case, True, "highway")

    # 3. 收费站 / 入口 / 出口通行
    # 注意：不再生成 “收费站堵不堵” 这种当前 v1 不稳的正样本
    node_templates = [
        "{x}{q}",
        "{x}现在{q}",
        "{x}今天{q}",
        "{x}还{q}",
    ]
    node_queries = [
        "能上吗", "开着吗", "封没封", "封了吗", "通了吗",
        "能下吗", "能不能上", "能不能下", "可以上吗", "可以下吗"
    ]

    def build_node_case():
        x = random.choice(toll_stations + entrances + exits)
        q = random.choice(node_queries)
        return wrap(random.choice(node_templates).format(x=x, q=q))

    true_cases += fill_cases(110, build_node_case, True, "node")

    # 4. 路段 / 主线 / 匝道 / 枢纽
    # 不生成 bare “省道咋样”“国道通不通”
    section_templates = [
        "前方{s}{q}",
        "后方{s}{q}",
        "这段{s}{q}",
        "{s}现在{q}",
        "{s}是不是{e}",
        "{s}有没有{e2}",
        "{s}什么情况",
        "{s}咋样",
    ]

    def build_section_case():
        s = random.choice(road_sections)
        q = random.choice(["堵不堵", "堵吗", "怎么样", "好走吗", "通不通", "能走吗"])
        e = random.choice(["堵车了", "排队了", "出事故了", "施工了", "封了"])
        e2 = random.choice(["事故", "施工", "管制", "分流", "缓行"])
        return wrap(random.choice(section_templates).format(s=s, q=q, e=e, e2=e2))

    true_cases += fill_cases(90, build_section_case, True, "section")

    # 5. 事件直接问法
    event_templates = [
        "{e}",
        "前方{e}",
        "后方{e}",
        "现在{e}",
        "{loc}{e}",
    ]
    loc_pool = highways + toll_stations + entrances + exits + road_sections

    def build_event_case():
        e = random.choice(traffic_events)
        loc = random.choice(loc_pool)
        return wrap(random.choice(event_templates).format(e=e, loc=loc))

    true_cases += fill_cases(60, build_event_case, True, "event")

    # 6. 直接路况问法
    def build_direct_case():
        return wrap(random.choice(traffic_direct))

    true_cases += fill_cases(40, build_direct_case, True, "direct")

    # -----------------------------
    # 负样本生成
    # -----------------------------
    false_cases = []

    negative_pool = negative_weather + negative_order + negative_review + negative_misc

    def build_plain_negative():
        return wrap(random.choice(negative_pool))

    false_cases += fill_cases(220, build_plain_negative, False, "plain_negative")

    # 更像 traffic 的 hard negative
    false_tricky_templates = [
        "帮我看下{obj}",
        "看看{obj}怎么样",
        "{obj}什么情况",
        "{obj}咋样",
        "{obj}正常吗",
        "{obj}可以吗",
        "{obj}多久能到",
    ]
    false_objs = [
        "订单", "物流", "快递", "外卖", "方案", "代码", "简历", "论文",
        "天气", "基金", "股票", "房间", "酒店", "快件", "面试结果", "作业"
    ]

    def build_hard_negative():
        tpl = random.choice(false_tricky_templates)
        obj = random.choice(false_objs)
        return wrap(tpl.format(obj=obj))

    false_cases += fill_cases(120, build_hard_negative, False, "hard_negative")

    # 以前会当 traffic，但当前 v1 明确不覆盖的“语义交通句”
    def build_unsupported_case():
        return wrap(random.choice(unsupported_semantic_traffic))

    false_cases += fill_cases(false_target - len(false_cases), build_unsupported_case, False, "unsupported_scope")

    # -----------------------------
    # 合并
    # -----------------------------
    dataset = true_cases[:true_target] + false_cases[:false_target]
    random.shuffle(dataset)

    return [
        {"text": text, "label": label, "tag": tag}
        for text, label, tag in dataset
    ]


if __name__ == "__main__":
    tests = build_traffic_testset_v1_aligned(total=1000)

    print("total:", len(tests))
    print("true:", sum(1 for x in tests if x["label"]))
    print("false:", sum(1 for x in tests if not x["label"]))

    correct = 0
    tp = tn = fp = fn = 0
    bad_cases = []

    tag_stats = {}

    for t in tests:
        text = t["text"]
        label = t["label"]
        tag = t["tag"]
        pred = looks_like_traffic_query_v1(text)

        if pred == label:
            correct += 1
        else:
            bad_cases.append({
                "text": text,
                "label": label,
                "pred": pred,
                "tag": tag,
            })

        if pred is True and label is True:
            tp += 1
        elif pred is False and label is False:
            tn += 1
        elif pred is True and label is False:
            fp += 1
        elif pred is False and label is True:
            fn += 1

        if tag not in tag_stats:
            tag_stats[tag] = {"total": 0, "correct": 0}
        tag_stats[tag]["total"] += 1
        if pred == label:
            tag_stats[tag]["correct"] += 1

    accuracy = correct / len(tests) if tests else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n===== 评估结果 =====")
    print(f"准确率   : {accuracy:.4f}")
    print(f"精确率   : {precision:.4f}")
    print(f"召回率   : {recall:.4f}")
    print(f"F1 值    : {f1:.4f}")

    print("\n===== 混淆矩阵 =====")
    print(f"真正例 TP（预测为路况，实际为路况）     : {tp}")
    print(f"真负例 TN（预测为非路况，实际为非路况） : {tn}")
    print(f"假正例 FP（预测为路况，实际为非路况）   : {fp}")
    print(f"假负例 FN（预测为非路况，实际为路况）   : {fn}")

    print("\n===== 分类统计 =====")
    tag_name_map = {
        "hard_negative": "高混淆负样本",
        "highway": "高速路况",
        "node": "收费站/出入口",
        "od": "OD 问法",
        "plain_negative": "普通负样本",
        "section": "路段/主线/匝道",
        "unsupported_scope": "当前规则未覆盖样本",
    }

    for tag, s in sorted(tag_stats.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        zh_tag = tag_name_map.get(tag, tag)
        print(f"{zh_tag:18s} 样本数={s['total']:4d}  准确率={acc:.4f}")

    print("\n===== Bad Cases (top 50) =====")
    for x in bad_cases[:50]:
        print(x)