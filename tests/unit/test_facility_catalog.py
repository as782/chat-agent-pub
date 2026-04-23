from __future__ import annotations

from app.agent.facility_catalog import FacilityCatalog


def test_service_area_match_prefers_normalized_group_name() -> None:
    catalog = FacilityCatalog.from_raw_rows(
        service_rows=[
            {
                "name": "嵊州服务区(常台)",
                "geographical_zone": "东区",
                "road_id": "33161",
                "road_gb_code": "G1522",
                "road_name": "G1522常台高速（上三）",
                "direction_type": "02",
                "direction": "下行",
                "direction_name": "绍兴方向",
                "direction_aliasname": "绍向",
                "start_site": "台州",
                "end_site": "绍兴",
            },
            {
                "name": "嵊州服务区(常台)",
                "geographical_zone": "西区",
                "road_id": "33161",
                "road_gb_code": "G1522",
                "road_name": "G1522常台高速（上三）",
                "direction_type": "01",
                "direction": "上行",
                "direction_name": "台州方向",
                "direction_aliasname": "台向",
                "start_site": "绍兴",
                "end_site": "台州",
            },
        ]
    )

    matches = catalog.match_service_area("嵊州服务区怎么样")

    assert matches
    assert matches[0].record.canonical_name == "嵊州服务区(常台)"
    assert matches[0].record.group_name == "嵊州服务区"
    assert catalog.best_service_keyword("嵊州服务区怎么样") == "嵊州服务区"
    assert "嵊州服务区" in catalog.resolve_service_query_terms("嵊州服务区怎么样")


def test_toll_station_match_disambiguates_by_road_code() -> None:
    catalog = FacilityCatalog.from_raw_rows(
        toll_rows=[
            {
                "facility_id": "1031",
                "name": "机场收费站",
                "road_id": "33141",
                "road_name": "G92杭州湾环线高速（杭甬）",
                "road_gb_code": "G92",
            },
            {
                "facility_id": "3463",
                "name": "机场收费站",
                "road_id": "34241",
                "road_name": "G1523甬莞高速（温州段）",
                "road_gb_code": "G1523",
            },
        ]
    )

    matches = catalog.match_toll_station("G1523机场收费站")

    assert matches
    assert matches[0].record.canonical_name == "机场收费站"
    assert matches[0].record.road_code == "G1523"
    assert matches[0].record.road_name_core == "甬莞高速（温州段）"
    assert catalog.best_toll_station("G1523机场收费站").road_code == "G1523"


def test_toll_station_match_respects_station_kind_aliases() -> None:
    catalog = FacilityCatalog.from_raw_rows(
        toll_rows=[
            {
                "facility_id": "1013",
                "name": "嘉善收费主站",
                "road_id": "33102",
                "road_name": "G60沪昆高速（沪杭）",
                "road_gb_code": "G60",
            }
        ]
    )

    matches = catalog.match_toll_station("嘉善主站")

    assert matches
    assert matches[0].record.group_name == "嘉善"
    assert matches[0].record.station_kind == "收费主站"
    assert "嘉善收费站" in matches[0].record.preferred_query_terms


def test_toll_station_match_resolves_road_code_for_canonical_station() -> None:
    catalog = FacilityCatalog.load_default()

    match = catalog.best_toll_station("嘉善收费主站")

    assert match is not None
    assert match.canonical_name == "嘉善收费主站"
    assert match.road_code == "G60"


def test_service_area_match_prefers_more_specific_prefix_name() -> None:
    catalog = FacilityCatalog.load_default()

    assert catalog.best_service_keyword("神仙居服务区状况如何", source="test") == "神仙居服务区"
