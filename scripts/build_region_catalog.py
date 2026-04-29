"""从 GB2260 TSV 数据生成 OD 行政区划字典。

生成后的 JSON 在运行时直接被地点索引加载，不需要线上访问外部数据源。
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

DEFAULT_SOURCE_URL = "https://raw.githubusercontent.com/cn/GB2260/develop/mca/201904.tsv"
DEFAULT_OUTPUT_PATH = Path("app/agent/data/region_catalog.json")

# 直辖市在 GB2260 中是省级代码，但 OD 语义上更接近城市目的地。
DIRECT_CITIES = {"北京市", "天津市", "上海市", "重庆市"}
SPECIAL_ADMINISTRATIVE_REGIONS = {"香港特别行政区", "澳门特别行政区"}
AUTONOMOUS_REGIONS = {
    "内蒙古自治区",
    "广西壮族自治区",
    "西藏自治区",
    "宁夏回族自治区",
    "新疆维吾尔自治区",
}

# 生成 canonical_name 时剥离的行政区划后缀。
# 例如“杭州市”生成“杭州”，“萧山区”生成“萧山”。
CANONICAL_SUFFIXES = (
    "特别行政区",
    "维吾尔自治区",
    "壮族自治区",
    "回族自治区",
    "自治区",
    "省",
    "市",
    "地区",
    "盟",
    "自治州",
    "自治县",
    "自治旗",
    "矿区",
    "林区",
    "新区",
    "区",
    "县",
    "旗",
)


def main() -> None:
    """读取源 TSV 并写出项目运行时使用的 region_catalog.json。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    rows = _read_tsv(args.source_url)
    code_to_name = {row["Code"]: row["Name"] for row in rows}
    regions = [
        _build_region(row["Code"], row["Name"], code_to_name)
        for row in rows
        if _should_include(row["Code"], row["Name"])
    ]
    payload = {
        "metadata": {
            "source": "cn/GB2260",
            "source_url": args.source_url,
            "scope": "county_level_and_above",
            "generated_by": "scripts/build_region_catalog.py",
        },
        "regions": regions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} with {len(regions)} regions")


def _read_tsv(source_url: str) -> list[dict[str, str]]:
    """读取 GB2260 TSV 源数据。"""

    with urllib.request.urlopen(source_url, timeout=30) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def _should_include(code: str, name: str) -> bool:
    """过滤不适合作为用户 OD 地点的占位行。"""

    if not code or not name:
        return False
    # “市辖区”是层级占位，不是用户自然会输入的 OD 地点。
    return name != "市辖区"


def _build_region(code: str, name: str, code_to_name: dict[str, str]) -> dict[str, object]:
    """把一行 GB2260 记录转换成运行时 region 结构。"""

    canonical_name = _canonical_name(name)
    return {
        "canonical_name": canonical_name,
        "place_type": _place_type(code, name),
        "parent": _parent_name(code, code_to_name),
        "adcode": code,
        "aliases": _aliases(name, canonical_name),
    }


def _place_type(code: str, name: str) -> str:
    """根据行政区划代码和名称推断地点层级类型。"""

    if name in SPECIAL_ADMINISTRATIVE_REGIONS:
        return "special_administrative_region"
    if name in AUTONOMOUS_REGIONS:
        return "autonomous_region"
    if code.endswith("0000"):
        return "municipality" if name in DIRECT_CITIES else "province"
    if code.endswith("00"):
        if name.endswith("自治州"):
            return "autonomous_prefecture"
        if name.endswith("地区"):
            return "prefecture"
        if name.endswith("盟"):
            return "league"
        return "city"
    if name.endswith("自治县"):
        return "autonomous_county"
    if name.endswith("自治旗"):
        return "autonomous_banner"
    if name.endswith("旗"):
        return "banner"
    if name.endswith("市"):
        return "county_level_city"
    if name.endswith("区"):
        return "district"
    if name.endswith("县"):
        return "county"
    return "region"


def _parent_name(code: str, code_to_name: dict[str, str]) -> str:
    """推断上级行政区名称，供候选日志和后续消歧使用。"""

    if code.endswith("0000"):
        return "中国"
    if code.endswith("00"):
        province_code = f"{code[:2]}0000"
        return _canonical_name(code_to_name.get(province_code, "中国"))
    city_code = f"{code[:4]}00"
    province_code = f"{code[:2]}0000"
    parent = code_to_name.get(city_code) or code_to_name.get(province_code) or "中国"
    return _canonical_name(parent)


def _canonical_name(name: str) -> str:
    """生成适合用户口语输入匹配的短名称。"""

    for suffix in CANONICAL_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _aliases(name: str, canonical_name: str) -> list[str]:
    """生成别名列表，保留全称和短名。"""

    aliases = [name]
    if canonical_name != name:
        aliases.append(canonical_name)
    return list(dict.fromkeys(alias for alias in aliases if alias))


if __name__ == "__main__":
    main()
