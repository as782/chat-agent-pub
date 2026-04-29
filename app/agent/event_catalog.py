"""Shared event type catalogs for live-agent event payloads."""

from __future__ import annotations

from typing import TypedDict


class SubEventType(TypedDict):
    event_type_id: str
    name: str


EVENT_TYPE_NAMES: dict[str, str] = {
    "01": "交通事件",
    "02": "交通灾害",
    "03": "交通气象",
    "04": "路面状况",
    "05": "路面施工",
    "06": "活动",
    "07": "重大事件",
    "09": "其他",
    "97": "车辆故障",
    "98": "服务区事件",
    "99": "收费站入口关闭",
    "100": "收费站入口限流",
    "101": "收费站出口关闭",
    "103": "主线管制",
    "104": "收费站出口分流",
    "105": "道路缓行",
}


SUB_EVENT_TYPES: dict[str, SubEventType] = {
    "010201": {"event_type_id": "01", "name": "撞行人"},
    "010202": {"event_type_id": "01", "name": "人车坠落"},
    "010301": {"event_type_id": "01", "name": "追尾"},
    "010302": {"event_type_id": "01", "name": "刮擦"},
    "010303": {"event_type_id": "01", "name": "翻车"},
    "010400": {"event_type_id": "01", "name": "其他设施相关"},
    "010401": {"event_type_id": "01", "name": "撞固定物"},
    "010402": {"event_type_id": "01", "name": "船舶撞桥"},
    "019600": {"event_type_id": "01", "name": "车辆起火"},
    "019700": {"event_type_id": "01", "name": "撞动物"},
    "019800": {"event_type_id": "01", "name": "撞抛洒物"},
    "019900": {"event_type_id": "01", "name": "其他"},
    "020000": {"event_type_id": "02", "name": "无"},
    "020200": {"event_type_id": "02", "name": "路面火灾"},
    "020300": {"event_type_id": "02", "name": "路边火灾"},
    "020400": {"event_type_id": "02", "name": "隧道火灾"},
    "020500": {"event_type_id": "02", "name": "道路设施火灾"},
    "020600": {"event_type_id": "02", "name": "其他地质灾害"},
    "020601": {"event_type_id": "02", "name": "山体滑坡"},
    "020602": {"event_type_id": "02", "name": "桥梁损坏"},
    "020603": {"event_type_id": "02", "name": "道路损坏"},
    "020604": {"event_type_id": "02", "name": "隧道塌方"},
    "020700": {"event_type_id": "02", "name": "水灾"},
    "029800": {"event_type_id": "02", "name": "环境污染"},
    "029900": {"event_type_id": "02", "name": "其他"},
    "030000": {"event_type_id": "03", "name": "无"},
    "030100": {"event_type_id": "03", "name": "大雨"},
    "030200": {"event_type_id": "03", "name": "冰雹"},
    "030300": {"event_type_id": "03", "name": "雷电"},
    "030400": {"event_type_id": "03", "name": "大风"},
    "030500": {"event_type_id": "03", "name": "雾霾"},
    "030600": {"event_type_id": "03", "name": "高温"},
    "030700": {"event_type_id": "03", "name": "干旱"},
    "030900": {"event_type_id": "03", "name": "寒潮"},
    "031000": {"event_type_id": "03", "name": "霜冻"},
    "039700": {"event_type_id": "03", "name": "雪"},
    "039800": {"event_type_id": "03", "name": "台风"},
    "039900": {"event_type_id": "03", "name": "其他"},
    "040000": {"event_type_id": "04", "name": "无"},
    "040100": {"event_type_id": "04", "name": "其他散乱物体"},
    "040101": {"event_type_id": "04", "name": "抛洒物"},
    "040102": {"event_type_id": "04", "name": "货物倾斜"},
    "040103": {"event_type_id": "04", "name": "货物散落"},
    "040104": {"event_type_id": "04", "name": "摩托车"},
    "040300": {"event_type_id": "04", "name": "机油泄漏"},
    "040500": {"event_type_id": "04", "name": "人"},
    "040600": {"event_type_id": "04", "name": "动物"},
    "040700": {"event_type_id": "04", "name": "积水"},
    "040800": {"event_type_id": "04", "name": "湿滑"},
    "040900": {"event_type_id": "04", "name": "道路结冰"},
    "049500": {"event_type_id": "04", "name": "倒车"},
    "049600": {"event_type_id": "04", "name": "停车"},
    "049700": {"event_type_id": "04", "name": "逆行"},
    "049800": {"event_type_id": "04", "name": "非机动车"},
    "049900": {"event_type_id": "04", "name": "其他"},
    "049901": {"event_type_id": "04", "name": "隐患点预警"},
    "050000": {"event_type_id": "05", "name": "无"},
    "050101": {"event_type_id": "05", "name": "日常养护（占道）"},
    "050102": {"event_type_id": "05", "name": "专项工程（占道）"},
    "050103": {"event_type_id": "05", "name": "临时抢修（占道）"},
    "050201": {"event_type_id": "05", "name": "日常养护（断路）"},
    "050202": {"event_type_id": "05", "name": "专项工程（断路）"},
    "050203": {"event_type_id": "05", "name": "临时抢修（断路）"},
    "050301": {"event_type_id": "05", "name": "专项工程（借道）"},
    "050302": {"event_type_id": "05", "name": "临时抢修（借道）"},
    "050401": {"event_type_id": "05", "name": "拓宽施工"},
    "059900": {"event_type_id": "05", "name": "其他"},
    "060000": {"event_type_id": "06", "name": "无"},
    "060100": {"event_type_id": "06", "name": "文体商业活动"},
    "060200": {"event_type_id": "06", "name": "外交政务活动"},
    "069900": {"event_type_id": "06", "name": "其他"},
    "070000": {"event_type_id": "07", "name": "无"},
    "070100": {"event_type_id": "07", "name": "燃气事故"},
    "070200": {"event_type_id": "07", "name": "化学污染"},
    "070201": {"event_type_id": "07", "name": "危化品事故"},
    "070300": {"event_type_id": "07", "name": "核事故"},
    "070400": {"event_type_id": "07", "name": "爆炸"},
    "070500": {"event_type_id": "07", "name": "电力事故"},
    "070600": {"event_type_id": "07", "name": "公共暴力"},
    "070601": {"event_type_id": "07", "name": "恶意事件"},
    "070602": {"event_type_id": "07", "name": "群体事件"},
    "070700": {"event_type_id": "07", "name": "交通集中堵塞"},
    "070701": {"event_type_id": "07", "name": "大流量"},
    "079800": {"event_type_id": "07", "name": "警卫任务"},
    "079900": {"event_type_id": "07", "name": "其他"},
    "090000": {"event_type_id": "09", "name": "无"},
    "099700": {"event_type_id": "09", "name": "内部管理"},
    "099800": {"event_type_id": "09", "name": "协助处理"},
    "099900": {"event_type_id": "09", "name": "其他"},
    "970100": {"event_type_id": "97", "name": "抛锚"},
    "970200": {"event_type_id": "97", "name": "爆胎"},
    "979900": {"event_type_id": "97", "name": "其他"},
    "980000": {"event_type_id": "98", "name": "无"},
    "980100": {"event_type_id": "98", "name": "缺油"},
    "980200": {"event_type_id": "98", "name": "无停车位"},
    "980300": {"event_type_id": "98", "name": "服务区关闭"},
    "980400": {"event_type_id": "98", "name": "服务区拥堵"},
    "989900": {"event_type_id": "98", "name": "其他"},
}


def resolve_event_type_name(value: object | None) -> str | None:
    """Resolve an event type id to its Chinese label."""

    code = _normalize_code(value)
    if code is None:
        return None
    return EVENT_TYPE_NAMES.get(code)


def resolve_sub_event_type(value: object | None) -> SubEventType | None:
    """Resolve a sub-event type id to its parent id and Chinese label."""

    code = _normalize_code(value)
    if code is None:
        return None
    return SUB_EVENT_TYPES.get(code)


def _normalize_code(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
