"""Deterministic renderer for network report markdown tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from logging import Logger
import re

from app.agent.event_filter import should_filter_report_event
from app.core.logger import get_logger
from app.agent.state import ExecutorResult

_ROAD_CODE_PATTERN = re.compile(r"([GS]\d{1,4})", re.IGNORECASE)
_ROAD_NAME_PREFIX_PATTERN = re.compile(r"^[GS]\d{1,4}", re.IGNORECASE)
_PAREN_SEGMENT_PATTERN = re.compile(r"[（(](.*?)[）)]")
_SUFFIX_SEGMENT_PATTERN = re.compile(r"^(.*?)([\u4e00-\u9fffA-Za-z0-9\[\]]{1,12}段?)$")

_TABLE_HEADER = "| roadCode | highwayName | roadSection | controls | traffic |"
_TABLE_SEPARATOR = "| --- | --- | --- | --- | --- |"
LOGGER: Logger = get_logger(__name__)


@dataclass(slots=True)
class _RoadProfile:
    code: str
    highway_name: str
    segment: str
    raw_name: str
    score: int


@dataclass(slots=True)
class _ReportRow:
    road_code: str
    highway_name: str
    segment: str
    station_control: str
    traffic: str


@dataclass(slots=True)
class RenderedNetworkReport:
    summary: str
    table_markdown: str
    rows: list[_ReportRow]

    def to_markdown(self) -> str:
        return f"{self.summary}\n\n{self.table_markdown}"


def _should_filter_event(item: dict[str, object]) -> bool:
    """判断是否应该过滤掉该事件，基于事件分类码值。
    
    根据需求，需要过滤掉交通事故(04)和车辆故障(07)等事件，
    以及10110硬路肩开放的管制类型。
    """
    return should_filter_report_event(item)


def build_network_report_render_result(
    step_results: dict[str, object],
) -> RenderedNetworkReport | None:
    """Build the stable table and deterministic fallback summary."""

    report_result = _find_report_result(step_results)
    if report_result is None:
        return None

    normalized_result = report_result.normalized_result or {}
    if not isinstance(normalized_result, dict):
        return None

    congestion_items = _coerce_items(normalized_result.get("congestion_top_items"))
    control_items = _coerce_items(normalized_result.get("control_top_items"))
    exit_items = _coerce_items(normalized_result.get("exit_top_items"))

    # 对拥堵事件进行筛选，过滤掉不需要的事件类型
    filtered_congestion_items = [
        item for item in congestion_items 
        if not _should_filter_event(item)
    ]
    
    # 对主线管制事件进行筛选
    filtered_control_items = [
        item for item in control_items 
        if not _should_filter_event(item)
    ]
    
    # 收费站管制通常不需要过滤，保持原样
    filtered_exit_items = exit_items

    road_profiles = _build_road_profiles(
        congestion_items=filtered_congestion_items,
        control_items=filtered_control_items,
        exit_items=filtered_exit_items,
    )

    exit_rows = _build_exit_rows(exit_items=filtered_exit_items, road_profiles=road_profiles)
    control_rows = _build_control_rows(control_items=filtered_control_items, road_profiles=road_profiles)
    congestion_rows = _build_congestion_rows(
        congestion_items=filtered_congestion_items,
        road_profiles=road_profiles,
    )
    rows = [*exit_rows, *control_rows, *congestion_rows]

    summary = _build_summary(
        exit_row_count=len(exit_rows),
        control_row_count=len(control_rows),
        congestion_row_count=len(congestion_rows),
        congestion_total_mile=normalized_result.get("congestion_total_mile"),
    )

    table_lines = [_TABLE_HEADER, _TABLE_SEPARATOR]
    if not rows:
        table_lines.append("| 无 | 无 | 无 | 无 | 无 |")
    else:
        for row in rows:
            table_lines.append(
                "| {road_code} | {highway_name} | {segment} | {station_control} | {traffic} |".format(
                    road_code=_escape_markdown_cell(row.road_code),
                    highway_name=_escape_markdown_cell(row.highway_name),
                    segment=_escape_markdown_cell(row.segment),
                    station_control=_escape_markdown_cell(row.station_control),
                    traffic=_escape_markdown_cell(row.traffic),
                )
            )

    rendered_report = RenderedNetworkReport(
        summary=summary,
        table_markdown="\n".join(table_lines),
        rows=rows,
    )
    LOGGER.info(
        "Network report table generated: rows=%s\nsummary=%s\ntable=\n%s",
        len(rendered_report.rows),
        rendered_report.summary,
        rendered_report.table_markdown,
    )
    return rendered_report


def render_network_report_from_step_results(
    step_results: dict[str, object],
) -> str | None:
    """Render a stable markdown report from report step results."""

    render_result = build_network_report_render_result(step_results)
    if render_result is None:
        return None
    return render_result.to_markdown()


def coerce_executor_result(step_id: str, result: object) -> ExecutorResult | None:
    """Normalize a step result coming from graph state or serialized event output."""

    if isinstance(result, ExecutorResult):
        return result
    if not isinstance(result, Mapping):
        return None

    executor = str(result.get("executor") or "").strip()
    if not executor:
        return None
    is_success = bool(result.get("is_success"))
    if not is_success:
        return None

    raw_result = result.get("raw_result")
    normalized_result = result.get("normalized_result")
    summary = result.get("summary")
    sources = result.get("sources")
    error = result.get("error")

    return ExecutorResult(
        step_id=str(result.get("step_id") or step_id),
        executor=executor,  # type: ignore[arg-type]
        is_success=is_success,
        raw_result=dict(raw_result) if isinstance(raw_result, Mapping) else {},
        normalized_result=dict(normalized_result) if isinstance(normalized_result, Mapping) else {},
        summary=summary if isinstance(summary, str) and summary.strip() else None,
        sources=[str(item) for item in sources if isinstance(item, str)] if isinstance(sources, list) else [],
        error=error if isinstance(error, str) and error.strip() else None,
    )


def _find_report_result(step_results: dict[str, object]) -> ExecutorResult | None:
    for step_id, result in step_results.items():
        normalized_result = coerce_executor_result(step_id, result)
        if (
            normalized_result is not None
            and normalized_result.executor == "report"
            and normalized_result.is_success
        ):
            return normalized_result
    return None


def _coerce_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _build_road_profiles(
    *,
    congestion_items: list[dict[str, object]],
    control_items: list[dict[str, object]],
    exit_items: list[dict[str, object]],
) -> dict[str, _RoadProfile]:
    profiles: dict[str, _RoadProfile] = {}
    for item in [*exit_items, *control_items, *congestion_items]:
        road_key = _build_road_key(item)
        profile = _parse_road_profile(item)
        existing_profile = profiles.get(road_key)
        if existing_profile is None or profile.score > existing_profile.score:
            profiles[road_key] = profile
    return profiles


def _build_exit_rows(
    *,
    exit_items: list[dict[str, object]],
    road_profiles: dict[str, _RoadProfile],
) -> list[_ReportRow]:
    grouped_rows: dict[tuple[str, str, str, str, str], dict[str, object]] = {}

    for item in exit_items:
        road_profile = _resolve_road_profile(item, road_profiles)
        station_name = _first_non_empty(
            item.get("tollName"),
            item.get("exitName"),
            item.get("entranceName"),
        ) or "未知收费站"
        direction = _format_direction(
            item.get("directionName"),
            item.get("directionType"),
            item.get("direction"),
        )
        group_key = (
            road_profile.code,
            road_profile.highway_name,
            road_profile.segment,
            station_name,
            direction,
        )
        row_group = grouped_rows.get(group_key)
        if row_group is None:
            row_group = {
                "profile": road_profile,
                "station_name": station_name,
                "direction": direction,
                "actions": [],
                "action_keys": set(),
            }
            grouped_rows[group_key] = row_group

        entrance_label = _format_entrance_label(
            item.get("entrance"),
            item.get("entranceName"),
        )
        action_text = _build_exit_action(item, entrance_label=entrance_label)
        action_key = (_entrance_order(item.get("entrance")), action_text)
        if action_key in row_group["action_keys"]:
            continue
        row_group["action_keys"].add(action_key)
        row_group["actions"].append(action_key)

    rows: list[_ReportRow] = []
    for row_group in grouped_rows.values():
        ordered_actions = [
            action_text
            for _, action_text in sorted(
                row_group["actions"],
                key=lambda action: (action[0], action[1]),
            )
        ]
        direction_text = str(row_group["direction"])
        action_suffix = "、".join(ordered_actions) if ordered_actions else "无"
        station_control = f"{row_group['station_name']}，{direction_text}{action_suffix}"
        profile = row_group["profile"]
        rows.append(
            _ReportRow(
                road_code=profile.code,
                highway_name=profile.highway_name,
                segment=profile.segment,
                station_control=station_control,
                traffic="无",
            )
        )
    return rows


def _build_control_rows(
    *,
    control_items: list[dict[str, object]],
    road_profiles: dict[str, _RoadProfile],
) -> list[_ReportRow]:
    rows: list[_ReportRow] = []
    for item in control_items:
        road_profile = _resolve_road_profile(item, road_profiles)
        direction = _format_direction(
            item.get("directionName"),
            item.get("directionType"),
            item.get("direction"),
        )
        control_type = _first_non_empty(
            item.get("controlTypeName"),
            _map_control_type(item.get("controlType")),
        ) or "管制"
        description = _normalize_control_reason(
            _first_non_empty(
                item.get("des"),
                item.get("limitMeasureTypeName"),
                item.get("controlMeasures"),
            )
        )
        traffic_parts = [direction, control_type]
        if description:
            traffic_parts.append(description)
        rows.append(
            _ReportRow(
                road_code=road_profile.code,
                highway_name=road_profile.highway_name,
                segment=road_profile.segment,
                station_control="无",
                traffic="，".join(part for part in traffic_parts if part and part != "未知"),
            )
        )
    return rows


def _build_congestion_rows(
    *,
    congestion_items: list[dict[str, object]],
    road_profiles: dict[str, _RoadProfile],
) -> list[_ReportRow]:
    rows: list[_ReportRow] = []
    for item in congestion_items:
        road_profile = _resolve_road_profile(item, road_profiles)
        direction = _format_direction(
            item.get("directionName"),
            item.get("directionType"),
            item.get("direction"),
        )
        description = _normalize_congestion_description(
            _first_non_empty(item.get("des")),
            road_profile=road_profile,
            direction=direction,
        )
        if not description:
 
            distance = _format_distance(item.get("roadAmbleMile"))
            # 调整顺序：方向在前，然后是位置和距离信息
            parts = []
            if direction and direction != "未知":
                # 将方向调整为"XX方向XX"的格式
                if "方向" in direction:
                    # 如果已经是"XX方向"格式，只需添加方向描述
                    parts.append(direction)
                else:
                    # 如果只是方向描述，加上"方向"
                    parts.append(f"{direction}方向")
            if distance and distance != "未知":
                parts.append(distance)
            description = "，".join(parts)
        rows.append(
            _ReportRow(
                road_code=road_profile.code,
                highway_name=road_profile.highway_name,
                segment=road_profile.segment,
                station_control="无",
                traffic=description or "无",
            )
        )
    return rows


def _build_summary(
    *,
    exit_row_count: int,
    control_row_count: int,
    congestion_row_count: int,
    congestion_total_mile: object | None,
) -> str:
    if exit_row_count == 0 and control_row_count == 0 and congestion_row_count == 0:
        return "当前全路网整体运行平稳，暂未监测到收费站管控、主线管制或缓行事件。"

    summary_parts: list[str] = []
    if exit_row_count:
        summary_parts.append(f"{exit_row_count}处收费站管控")
    if control_row_count:
        summary_parts.append(f"{control_row_count}处主线管制")
    if congestion_row_count:
        summary_parts.append(f"{congestion_row_count}处缓行事件")
    if congestion_total_mile is not None:
        summary_parts.append(f"拥堵总里程约{_format_numeric_value(congestion_total_mile)}公里")

    return "当前全路网监测到" + "、".join(summary_parts) + "，整体以局部异常为主。"


def _resolve_road_profile(
    item: dict[str, object],
    road_profiles: dict[str, _RoadProfile],
) -> _RoadProfile:
    return road_profiles.get(_build_road_key(item)) or _parse_road_profile(item)


def _parse_road_profile(item: dict[str, object]) -> _RoadProfile:
    raw_name = _first_non_empty(item.get("roadName")) or "未知"
    explicit_code = _first_non_empty(
        item.get("roadGBCode"),
        item.get("roadGbCode"),
        item.get("roadCode"),
    )
    road_code = explicit_code or _extract_road_code(raw_name) or "未知"

    name_without_code = _ROAD_NAME_PREFIX_PATTERN.sub("", raw_name).strip() or raw_name
    paren_match = _PAREN_SEGMENT_PATTERN.search(name_without_code)
    segment = paren_match.group(1).strip() if paren_match else None
    base_name = _PAREN_SEGMENT_PATTERN.sub("", name_without_code).strip(" -") or name_without_code

    if segment is None:
        suffix_match = _SUFFIX_SEGMENT_PATTERN.match(base_name)
        if suffix_match:
            candidate_name = suffix_match.group(1).strip()
            candidate_segment = suffix_match.group(2).strip()
            if candidate_name:
                base_name = candidate_name
                segment = candidate_segment

    highway_name = base_name.strip() or "未知"
    if (
        highway_name != "未知"
        and "高速" not in highway_name
        and "通道" not in highway_name
        and "环线" not in highway_name
        and "绕城" not in highway_name
    ):
        highway_name = f"{highway_name}高速"

    normalized_segment = segment.strip() if isinstance(segment, str) and segment.strip() else "无"
    score = 0
    if _extract_road_code(raw_name):
        score += 2
    if "高速" in raw_name or "通道" in raw_name:
        score += 2
    if normalized_segment != "无":
        score += 2
    if "（" in raw_name or "(" in raw_name:
        score += 1

    return _RoadProfile(
        code=road_code,
        highway_name=highway_name,
        segment=normalized_segment,
        raw_name=raw_name,
        score=score,
    )


def _build_road_key(item: dict[str, object]) -> str:
    explicit_code = _first_non_empty(
        item.get("roadGBCode"),
        item.get("roadGbCode"),
        item.get("roadCode"),
    )
    if explicit_code:
        return explicit_code

    raw_name = _first_non_empty(item.get("roadName"))
    extracted_code = _extract_road_code(raw_name)
    if extracted_code:
        return extracted_code

    road_id = _first_non_empty(item.get("roadId"), item.get("road"))
    if road_id:
        return road_id
    return raw_name or "unknown-road"


def _build_exit_action(item: dict[str, object], *, entrance_label: str) -> str:
    control_result = _first_non_empty(
        item.get("controlTypeName"),
        _map_control_type(item.get("controlType")),
    ) or "无"
    measure = _first_non_empty(item.get("limitMeasureTypeName"), item.get("controlMeasures"))
    if measure and measure not in control_result and measure != "无":
        control_result = f"{control_result}（{measure}）"
    return f"{entrance_label}{control_result}"


def _normalize_control_reason(description: str | None) -> str | None:
    if description is None:
        return None
    normalized_description = description.strip("。；;，, ")
    if not normalized_description:
        return None
 
    return normalized_description or None


def _normalize_congestion_description(
    description: str | None,
    *,
    road_profile: _RoadProfile,
    direction: str,
) -> str:
    if description is None:
        return ""
    normalized_description = description.strip().rstrip("。")
    if not normalized_description:
        return ""

    if normalized_description.startswith(road_profile.raw_name):
        normalized_description = normalized_description[len(road_profile.raw_name) :].lstrip("-—– ")
    if road_profile.code != "未知" and normalized_description.startswith(road_profile.code):
        normalized_description = normalized_description[len(road_profile.code) :].lstrip("-—– ")
    if direction != "未知" and normalized_description.startswith(direction):
        normalized_description = normalized_description[len(direction) :].lstrip("，, ")

    normalized_description = normalized_description.strip("，, ")
    if not normalized_description:
        return direction
 
    return normalized_description


def _format_location(begin_milestone: object | None, end_milestone: object | None) -> str:
    begin_text = _first_non_empty(begin_milestone)
    end_text = _first_non_empty(end_milestone)
    if not begin_text and not end_text:
        return "未知"
    if begin_text and end_text:
        return f"K{begin_text}-K{end_text}"
    return f"K{begin_text or end_text}"


def _format_distance(value: object | None) -> str:
    if value is None:
        return ""
    formatted_value = _format_numeric_value(value)
    if not formatted_value:
        return ""
    return f"缓行{formatted_value}公里"


def _format_numeric_value(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.1f}".rstrip("0").rstrip(".")


def _extract_road_code(value: object | None) -> str | None:
    if value is None:
        return None
    match = _ROAD_CODE_PATTERN.search(str(value))
    if match is None:
        return None
    return match.group(1).upper()


def _format_direction(*candidates: object | None) -> str:
    direction_map = {
        "0": "双向",
        "00": "双向",
        "双向": "双向",
        "1": "上行",
        "01": "上行",
        "100700": "上行",
        "上行": "上行",
        "2": "下行",
        "02": "下行",
        "100701": "下行",
        "下行": "下行",
    }
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        if text in direction_map:
            return direction_map[text]
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return text
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return "未知"


def _format_entrance_label(entrance: object | None, entrance_name: object | None) -> str:
    resolved_name = _first_non_empty(entrance_name)
    if resolved_name:
        return resolved_name
    entrance_text = _first_non_empty(entrance)
    entrance_map = {
        "0": "出口",
        "1": "入口",
        "出口": "出口",
        "入口": "入口",
    }
    if entrance_text is None:
        return "未知"
    return entrance_map.get(entrance_text, entrance_text)


def _map_control_type(value: object | None) -> str | None:
    control_type_map = {
        "10102": "单向封道",
        "10108": "枢纽匝道卡口",
        "10109": "主线限流",
        "10110": "开放硬路肩",
        "10202": "关闭",
        "10203": "限流",
        "10204": "分流",
    }
    if value is None:
        return None
    return control_type_map.get(str(value).strip())


def _entrance_order(value: object | None) -> int:
    normalized_value = _first_non_empty(value) or ""
    order_map = {"1": 0, "0": 1}
    return order_map.get(normalized_value, 9)


def _escape_markdown_cell(value: object | None) -> str:
    if value is None:
        return "无"
    text = str(value).strip()
    if not text:
        return "无"
    return text.replace("|", "｜").replace("\n", " ")


def _first_non_empty(*values: object | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
