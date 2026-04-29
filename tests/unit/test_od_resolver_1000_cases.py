from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from app.agent.od.resolver import OdResolver


@dataclass(frozen=True)
class OdCase:
    message: str
    origin: str
    destination: str
    category: str


ORIGINS = [
    "杭州", "上海", "宁波", "湖州", "嘉兴", "绍兴", "苏州", "南京", "无锡", "常州",
    "金华", "义乌", "台州", "温州", "丽水", "衢州", "舟山", "合肥", "芜湖", "黄山",
    "杭州东", "上海虹桥", "南京南", "宁波站", "嘉兴南", "湖州站", "苏州北", "无锡东",
    "杭州服务区", "嘉兴服务区", "湖州服务区", "绍兴服务区", "余姚服务区",
    "杭州南收费站", "萧山收费站", "嘉兴收费站", "湖州收费站", "宁波东收费站",
    "G60沪昆高速", "G25长深高速", "S2杭甬高速", "绕城西线", "申嘉湖高速",
]

DESTINATIONS = [
    "杭州", "上海", "宁波", "湖州", "嘉兴", "绍兴", "苏州", "南京", "无锡", "常州",
    "金华", "义乌", "台州", "温州", "丽水", "衢州", "舟山", "合肥", "芜湖", "黄山",
    "杭州东", "上海虹桥", "南京南", "宁波站", "嘉兴南", "湖州站", "苏州北", "无锡东",
    "杭州服务区", "嘉兴服务区", "湖州服务区", "绍兴服务区", "余姚服务区",
    "杭州南收费站", "萧山收费站", "嘉兴收费站", "湖州收费站", "宁波东收费站",
    "G60沪昆高速", "G25长深高速", "S2杭甬高速", "绕城西线", "申嘉湖高速",
]


TEMPLATES = [
    ("从{origin}到{destination}多久", "from_to_duration"),
    ("从{origin}去{destination}怎么走", "from_go_route"),
    ("从{origin}前往{destination}路况怎么样", "from_forward_traffic"),
    ("由{origin}至{destination}需要多长时间", "from_to_need_time"),
    ("{origin}到{destination}多久", "plain_to_duration"),
    ("{origin}去{destination}怎么走", "plain_go_route"),
    ("{origin}往{destination}方向堵不堵", "plain_direction_traffic"),
    ("{origin}前往{destination}收费多少", "plain_forward_fee"),
    ("{origin}至{destination}还有多远", "plain_to_distance"),
    ("{origin}回{destination}要多久", "plain_return_duration"),
    ("我现在在{origin}往{destination}方向还有多久", "current_at_direction"),
    ("我目前位于{origin}去{destination}怎么走", "current_located_go"),
    ("当前处在{origin}前往{destination}路况怎么样", "current_in_forward"),
    ("我已经到达{origin}去{destination}还要多久", "arrived_go"),
    ("现在在{origin}到{destination}需要多少时间", "current_at_to"),
    ("帮我看下从{origin}到{destination}收费多少", "polite_from_to_fee"),
    ("查一下{origin}到{destination}的路况", "query_traffic"),
    ("导航从{origin}去{destination}", "navigation_from_go"),
    ("请问{origin}到{destination}怎么走", "ask_plain_route"),
    ("想从{origin}开到{destination}，要多久", "drive_from_to"),
    ("{origin}开往{destination}方向路况如何", "drive_direction_traffic"),
    ("从{origin}回{destination}现在堵吗", "from_return_traffic"),
    ("我在{origin}，到{destination}多久", "comma_current_to"),
    ("我在{origin}，能到{destination}吗", "comma_current_can_arrive"),
    ("{origin}到{destination}过路费多少", "toll_fee"),
]


PREFIXES = [
    "",
    "你好，",
    "麻烦问一下，",
    "帮我看看，",
    "请帮我查下，",
    "现在",
    "今天",
    "这会儿",
]

SUFFIXES = [
    "",
    "？",
    "。",
    "，谢谢",
    "，急",
    "，现在出发",
    "，高速优先",
    "，避开拥堵",
    "，走高速",
    "，不要走小路",
]


def build_1000_od_cases(seed: int = 20260429, total: int = 1000) -> list[OdCase]:
    rng = random.Random(seed)
    cases: list[OdCase] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = total * 20

    while len(cases) < total and attempts < max_attempts:
        attempts += 1

        origin = rng.choice(ORIGINS)
        destination = rng.choice(DESTINATIONS)

        if origin == destination:
            continue

        template, category = rng.choice(TEMPLATES)
        prefix = rng.choice(PREFIXES)
        suffix = rng.choice(SUFFIXES)

        message = prefix + template.format(origin=origin, destination=destination) + suffix

        if message in seen:
            continue

        seen.add(message)
        cases.append(
            OdCase(
                message=message,
                origin=origin,
                destination=destination,
                category=category,
            )
        )

    assert len(cases) == total
    return cases


OD_1000_CASES = build_1000_od_cases()


@pytest.mark.parametrize("case", OD_1000_CASES, ids=lambda c: c.category)
def test_od_resolver_1000_cases(case: OdCase):
    result = OdResolver().resolve(case.message)

    assert result.origin == case.origin, (
        f"\nmessage={case.message}"
        f"\nexpected_origin={case.origin}"
        f"\nactual_origin={result.origin}"
        f"\nactual_destination={result.destination}"
        f"\nwarnings={result.warnings}"
        f"\nsource={result.source}"
    )

    assert result.destination == case.destination, (
        f"\nmessage={case.message}"
        f"\nexpected_destination={case.destination}"
        f"\nactual_origin={result.origin}"
        f"\nactual_destination={result.destination}"
        f"\nwarnings={result.warnings}"
        f"\nsource={result.source}"
    )


def test_od_resolver_1000_cases_accuracy_report():
    resolver = OdResolver()

    total = len(OD_1000_CASES)
    correct = 0
    failures = []

    by_category: dict[str, dict[str, int]] = {}

    for case in OD_1000_CASES:
        result = resolver.resolve(case.message)

        category_stat = by_category.setdefault(case.category, {"total": 0, "correct": 0})
        category_stat["total"] += 1

        is_correct = (
            result.origin == case.origin
            and result.destination == case.destination
        )

        if is_correct:
            correct += 1
            category_stat["correct"] += 1
        else:
            failures.append(
                {
                    "message": case.message,
                    "expected_origin": case.origin,
                    "expected_destination": case.destination,
                    "actual_origin": result.origin,
                    "actual_destination": result.destination,
                    "source": result.source,
                    "warnings": result.warnings,
                    "category": case.category,
                }
            )

    accuracy = correct / total

    print("\nOD resolver accuracy report")
    print(f"total={total}")
    print(f"correct={correct}")
    print(f"accuracy={accuracy:.2%}")

    print("\ncategory accuracy")
    for category, stat in sorted(by_category.items()):
        category_accuracy = stat["correct"] / stat["total"]
        print(
            f"{category}: "
            f"{stat['correct']}/{stat['total']} = {category_accuracy:.2%}"
        )

    if failures:
        print("\nfirst 30 failures")
        for item in failures[:30]:
            print(item)

    assert accuracy >= 0.95, f"accuracy={accuracy:.2%}, failures={len(failures)}"