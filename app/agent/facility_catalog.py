"""Facility catalog and fast lookup helpers for service areas and toll stations."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.core.logger import get_logger

RecordType = Literal["service_area", "toll_station"]

LOGGER = get_logger(__name__)

DEFAULT_CATALOG_PATH = Path(__file__).with_name("data") / "facility_catalog.json"

_NORMALIZE_PATTERN = re.compile(r"[\s\-_—–,，。\.、;；:：/\\|·`'\"“”‘’（）()【】\[\]{}<>《》]+")
_PAREN_PATTERN = re.compile(r"[（(][^（）()]*[）)]")
_ROAD_CODE_PATTERN = re.compile(r"^[GS]\d{1,4}$", re.IGNORECASE)


def _unique_strings(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = _NORMALIZE_PATTERN.sub("", text)
    return text


def _normalize_road_code(value: object | None) -> str:
    text = _normalize_text(value).upper()
    return text if _ROAD_CODE_PATTERN.fullmatch(text) else text


def _strip_parentheses(value: str) -> str:
    return _PAREN_PATTERN.sub("", value).strip()


def _road_name_core(value: object | None, road_code: str | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized_code = _normalize_road_code(road_code)
    if normalized_code and text.upper().startswith(normalized_code):
        text = text[len(normalized_code) :].strip()
    text = text.lstrip("：:").strip()
    return text


def _road_name_aliases(value: object | None, road_code: str | None = None) -> list[str]:
    raw_text = str(value or "").strip()
    core_text = _road_name_core(raw_text, road_code=road_code)
    aliases: list[str] = []
    if raw_text:
        aliases.append(raw_text)
    if core_text:
        aliases.append(core_text)
        aliases.append(_strip_parentheses(core_text))
        paren_match = re.search(r"[（(]([^（）()]+)[）)]", core_text)
        if paren_match is not None:
            aliases.append(paren_match.group(1).strip())
    if road_code:
        aliases.append(f"{road_code}{core_text}" if core_text else road_code)
    return list(_unique_strings([alias for alias in aliases if alias]))


def _build_search_terms(*values: object | None) -> tuple[str, ...]:
    terms: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                normalized = _normalize_text(item)
                if normalized:
                    terms.append(normalized)
            continue
        normalized = _normalize_text(value)
        if normalized:
            terms.append(normalized)
    return _unique_strings(terms)


@dataclass(frozen=True, slots=True)
class ServiceAreaRecord:
    """Normalized service area record."""

    record_id: str
    canonical_name: str
    group_name: str
    road_id: str
    road_code: str
    road_name_raw: str
    road_name_core: str
    geographical_zone: str
    direction_type: str
    direction: str
    direction_name: str
    direction_aliasname: str
    start_site: str
    end_site: str
    aliases: tuple[str, ...]
    preferred_query_terms: tuple[str, ...]
    search_terms: tuple[str, ...]
    group_key: str

    @classmethod
    def from_mapping(cls, mapping: dict[str, object], *, record_id: str | None = None) -> "ServiceAreaRecord":
        name = str(
            mapping.get("canonical_name")
            or mapping.get("name")
            or mapping.get("service_area_name")
            or ""
        ).strip()
        group_name = _strip_parentheses(name)
        road_code = _normalize_road_code(mapping.get("road_code") or mapping.get("road_gb_code"))
        road_name_raw = str(mapping.get("road_name_raw") or mapping.get("road_name") or "").strip()
        road_name_core = _road_name_core(road_name_raw, road_code=road_code)
        geographical_zone = str(mapping.get("geographical_zone") or mapping.get("zone") or "").strip()
        direction_type = str(mapping.get("direction_type") or "").strip()
        direction = str(mapping.get("direction") or "").strip()
        direction_name = str(mapping.get("direction_name") or "").strip()
        direction_aliasname = str(mapping.get("direction_aliasname") or "").strip()
        start_site = str(mapping.get("start_site") or "").strip()
        end_site = str(mapping.get("end_site") or "").strip()
        aliases = _unique_strings(
            [
                name,
                group_name,
                f"{group_name}{geographical_zone}" if group_name and geographical_zone else "",
                f"{group_name}{direction_name}" if group_name and direction_name else "",
                f"{group_name}{direction_aliasname}" if group_name and direction_aliasname else "",
                f"{group_name}{direction}" if group_name and direction else "",
                road_name_core,
                road_name_raw,
                start_site,
                end_site,
            ]
        )
        preferred_query_terms = _unique_strings(
            [
                f"{group_name}{geographical_zone}" if group_name and geographical_zone else "",
                f"{group_name}{direction_name}" if group_name and direction_name else "",
                f"{group_name}{direction_aliasname}" if group_name and direction_aliasname else "",
                group_name,
            ]
        )
        search_terms = _build_search_terms(
            aliases,
            road_code,
            road_name_core,
            road_name_raw,
            geographical_zone,
            direction_type,
            direction,
            direction_name,
            direction_aliasname,
            start_site,
            end_site,
        )
        group_key = f"{group_name}|{road_code or road_name_core}"
        return cls(
            record_id=record_id or str(mapping.get("record_id") or mapping.get("id") or len(group_name)),
            canonical_name=name,
            group_name=group_name,
            road_id=str(mapping.get("road_id") or "").strip(),
            road_code=road_code,
            road_name_raw=road_name_raw,
            road_name_core=road_name_core,
            geographical_zone=geographical_zone,
            direction_type=direction_type,
            direction=direction,
            direction_name=direction_name,
            direction_aliasname=direction_aliasname,
            start_site=start_site,
            end_site=end_site,
            aliases=aliases,
            preferred_query_terms=preferred_query_terms,
            search_terms=search_terms,
            group_key=group_key,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "canonical_name": self.canonical_name,
            "group_name": self.group_name,
            "road_id": self.road_id,
            "road_code": self.road_code,
            "road_name_raw": self.road_name_raw,
            "road_name_core": self.road_name_core,
            "geographical_zone": self.geographical_zone,
            "direction_type": self.direction_type,
            "direction": self.direction,
            "direction_name": self.direction_name,
            "direction_aliasname": self.direction_aliasname,
            "start_site": self.start_site,
            "end_site": self.end_site,
            "aliases": list(self.aliases),
            "preferred_query_terms": list(self.preferred_query_terms),
            "search_terms": list(self.search_terms),
            "group_key": self.group_key,
        }


@dataclass(frozen=True, slots=True)
class TollStationRecord:
    """Normalized toll station record."""

    record_id: str
    facility_id: str
    canonical_name: str
    group_name: str
    road_id: str
    road_code: str
    road_name_raw: str
    road_name_core: str
    station_kind: str
    aliases: tuple[str, ...]
    preferred_query_terms: tuple[str, ...]
    search_terms: tuple[str, ...]
    group_key: str

    @classmethod
    def from_mapping(cls, mapping: dict[str, object], *, record_id: str | None = None) -> "TollStationRecord":
        name = str(mapping.get("canonical_name") or mapping.get("name") or mapping.get("station_name") or "").strip()
        group_name = _strip_toll_suffix(name)
        road_code = _normalize_road_code(mapping.get("road_code") or mapping.get("road_gb_code"))
        road_name_raw = str(mapping.get("road_name_raw") or mapping.get("road_name") or "").strip()
        road_name_core = _road_name_core(road_name_raw, road_code=road_code)
        station_kind = _infer_station_kind(name)
        aliases = _unique_strings(
            [
                name,
                group_name,
                f"{group_name}收费站" if group_name else "",
                f"{group_name}收费主站" if group_name else "",
                f"{group_name}收费副站" if group_name else "",
                f"{group_name}主站" if group_name else "",
                f"{group_name}副站" if group_name else "",
                road_name_core,
                road_name_raw,
                road_code,
            ]
        )
        preferred_query_terms = _unique_strings(
            [
                name,
                f"{group_name}收费站" if group_name else "",
                f"{group_name}收费主站" if group_name else "",
                f"{group_name}收费副站" if group_name else "",
                f"{group_name}主站" if group_name else "",
                f"{group_name}副站" if group_name else "",
                group_name,
            ]
        )
        search_terms = _build_search_terms(
            aliases,
            road_code,
            road_name_core,
            road_name_raw,
            station_kind,
        )
        group_key = f"{group_name}|{road_code or road_name_core}"
        return cls(
            record_id=record_id or str(mapping.get("record_id") or mapping.get("facility_id") or len(group_name)),
            facility_id=str(mapping.get("facility_id") or mapping.get("id") or "").strip(),
            canonical_name=name,
            group_name=group_name,
            road_id=str(mapping.get("road_id") or "").strip(),
            road_code=road_code,
            road_name_raw=road_name_raw,
            road_name_core=road_name_core,
            station_kind=station_kind,
            aliases=aliases,
            preferred_query_terms=preferred_query_terms,
            search_terms=search_terms,
            group_key=group_key,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "facility_id": self.facility_id,
            "canonical_name": self.canonical_name,
            "group_name": self.group_name,
            "road_id": self.road_id,
            "road_code": self.road_code,
            "road_name_raw": self.road_name_raw,
            "road_name_core": self.road_name_core,
            "station_kind": self.station_kind,
            "aliases": list(self.aliases),
            "preferred_query_terms": list(self.preferred_query_terms),
            "search_terms": list(self.search_terms),
            "group_key": self.group_key,
        }


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Single ranked match result."""

    record_type: RecordType
    score: int
    reasons: tuple[str, ...]
    query_term: str
    record: ServiceAreaRecord | TollStationRecord

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "score": self.score,
            "reasons": list(self.reasons),
            "query_term": self.query_term,
            "record": self.record.to_dict(),
        }


@dataclass(slots=True)
class _RecordPool:
    records: tuple[ServiceAreaRecord | TollStationRecord, ...]
    term_index: dict[str, set[int]]


def _build_term_index(records: list[ServiceAreaRecord | TollStationRecord]) -> dict[str, set[int]]:
    term_index: dict[str, set[int]] = {}
    for idx, record in enumerate(records):
        for term in record.search_terms:
            term_index.setdefault(term, set()).add(idx)
    return term_index


def _collect_candidate_ids(query_norm: str, pool: _RecordPool) -> set[int]:
    candidate_ids: set[int] = set()
    if not query_norm:
        return candidate_ids
    for term, ids in pool.term_index.items():
        if term in query_norm or query_norm in term:
            candidate_ids.update(ids)
    return candidate_ids or set(range(len(pool.records)))


def _score_keyword_hits(query_norm: str, terms: tuple[str, ...], *, base_score: int) -> tuple[int, tuple[str, ...]]:
    score = 0
    reasons: list[str] = []
    for term in terms:
        if not term:
            continue
        if query_norm == term:
            score = max(score, base_score)
            reasons.append(f"exact:{term}")
            continue
        if term in query_norm:
            score = max(score, base_score - 150)
            reasons.append(f"contains:{term}")
            continue
        if query_norm in term:
            score = max(score, base_score - 250)
            reasons.append(f"within:{term}")
    return score, tuple(_unique_strings(reasons))


def _strip_toll_suffix(value: str) -> str:
    text = _strip_parentheses(value).strip()
    for suffix in ("收费主站", "收费副站", "收费站", "收费口", "主站", "副站"):
        if text.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


def _infer_station_kind(value: str) -> str:
    text = str(value or "").strip()
    if "收费主站" in text:
        return "收费主站"
    if "收费副站" in text:
        return "收费副站"
    if "收费口" in text:
        return "收费口"
    if "收费站" in text:
        return "收费站"
    if "入口" in text:
        return "入口"
    if "出口" in text:
        return "出口"
    return "收费站"


class FacilityCatalog:
    """Separate searchable catalogs for service areas and toll stations."""

    def __init__(
        self,
        *,
        service_areas: list[ServiceAreaRecord] | None = None,
        toll_stations: list[TollStationRecord] | None = None,
    ) -> None:
        self._service_areas = tuple(service_areas or [])
        self._toll_stations = tuple(toll_stations or [])
        self._service_pool = _RecordPool(
            records=self._service_areas,
            term_index=_build_term_index(list(self._service_areas)),
        )
        self._toll_pool = _RecordPool(
            records=self._toll_stations,
            term_index=_build_term_index(list(self._toll_stations)),
        )

    @classmethod
    def empty(cls) -> "FacilityCatalog":
        return cls()

    @classmethod
    def from_raw_rows(
        cls,
        *,
        service_rows: list[dict[str, object]] | None = None,
        toll_rows: list[dict[str, object]] | None = None,
    ) -> "FacilityCatalog":
        service_areas = [
            ServiceAreaRecord.from_mapping(row, record_id=str(index + 1))
            for index, row in enumerate(service_rows or [])
        ]
        toll_stations = [
            TollStationRecord.from_mapping(row, record_id=str(index + 1))
            for index, row in enumerate(toll_rows or [])
        ]
        return cls(service_areas=service_areas, toll_stations=toll_stations)

    @classmethod
    def from_json_payload(cls, payload: dict[str, object]) -> "FacilityCatalog":
        service_areas_payload = payload.get("service_areas", [])
        toll_stations_payload = payload.get("toll_stations", [])
        service_areas = [
            ServiceAreaRecord.from_mapping(item, record_id=str(index + 1))
            for index, item in enumerate(service_areas_payload)
            if isinstance(item, dict)
        ]
        toll_stations = [
            TollStationRecord.from_mapping(item, record_id=str(index + 1))
            for index, item in enumerate(toll_stations_payload)
            if isinstance(item, dict)
        ]
        return cls(service_areas=service_areas, toll_stations=toll_stations)

    @classmethod
    def load_default(cls) -> "FacilityCatalog":
        if not DEFAULT_CATALOG_PATH.exists():
            return cls.empty()
        payload = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return cls.empty()
        return cls.from_json_payload(payload)

    def to_json_payload(self) -> dict[str, object]:
        return {
            "service_areas": [record.to_dict() for record in self._service_areas],
            "toll_stations": [record.to_dict() for record in self._toll_stations],
        }

    def save_json(self, path: Path | str = DEFAULT_CATALOG_PATH) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_json_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def match_service_area(self, query: str, *, limit: int = 5) -> list[MatchResult]:
        return self._match_records(
            query=query,
            pool=self._service_pool,
            record_type="service_area",
            limit=limit,
        )

    def match_toll_station(self, query: str, *, limit: int = 5) -> list[MatchResult]:
        return self._match_records(
            query=query,
            pool=self._toll_pool,
            record_type="toll_station",
            limit=limit,
        )

    def best_service_keyword(self, query: str, *, source: str | None = None) -> str | None:
        started_at = time.perf_counter()
        query_norm = _normalize_text(query)
        matches = self.match_service_area(query, limit=5)
        if not matches:
            self._log_lookup(
                lookup_type="service",
                source=source,
                query=query,
                matches=[],
                started_at=started_at,
            )
            return None

        top_match = matches[0]
        accepted = self._is_strong_service_match(query_norm, top_match)
        self._log_lookup(
            lookup_type="service",
            source=source,
            query=query,
            matches=matches,
            started_at=started_at,
            accepted=accepted,
        )
        if not accepted:
            return None

        record = top_match.record
        if isinstance(record, ServiceAreaRecord):
            return record.group_name or record.canonical_name
        return None

    def best_toll_station(self, query: str, *, source: str | None = None) -> TollStationRecord | None:
        started_at = time.perf_counter()
        matches = self.match_toll_station(query, limit=3)
        if not matches:
            self._log_lookup(
                lookup_type="toll",
                source=source,
                query=query,
                matches=[],
                started_at=started_at,
            )
            return None
        record = matches[0].record
        self._log_lookup(
            lookup_type="toll",
            source=source,
            query=query,
            matches=matches,
            started_at=started_at,
        )
        return record if isinstance(record, TollStationRecord) else None

    def resolve_service_query_terms(self, query: str, *, limit: int = 3) -> list[str]:
        query_norm = _normalize_text(query)
        matches = self.match_service_area(query, limit=limit)
        strong_matches = [match for match in matches if self._is_strong_service_match(query_norm, match)]
        if not strong_matches:
            return []
        preferred_terms: list[str] = []
        for match in strong_matches:
            record = match.record
            if isinstance(record, ServiceAreaRecord):
                preferred_terms.extend(record.preferred_query_terms)
        return list(_unique_strings(preferred_terms))

    @staticmethod
    def _log_lookup(
        *,
        lookup_type: str,
        source: str | None,
        query: str,
        matches: list[MatchResult],
        started_at: float,
        accepted: bool | None = None,
    ) -> None:
        duration_ms = (time.perf_counter() - started_at) * 1000
        source_text = f" source={source}" if source else ""
        accepted_text = "" if accepted is None else f" accepted={str(accepted).lower()}"
        if not matches:
            LOGGER.info(
                "Facility catalog %s lookup%s: query=%s matched=None%s duration_ms=%.2f",
                lookup_type,
                source_text,
                query,
                accepted_text,
                duration_ms,
            )
            return

        top_match = matches[0]
        record = top_match.record
        if isinstance(record, TollStationRecord):
            record_name = record.canonical_name
            road_code = record.road_code or "-"
            road_name = record.road_name_core or "-"
            detail = f"record={record_name} road_code={road_code} road_name={road_name}"
        elif isinstance(record, ServiceAreaRecord):
            record_name = record.canonical_name
            road_code = record.road_code or "-"
            road_name = record.road_name_core or "-"
            detail = f"record={record_name} road_code={road_code} road_name={road_name}"
        else:
            detail = "record=unknown"
        LOGGER.info(
            "Facility catalog %s lookup%s: query=%s matched=%s score=%s %s%s duration_ms=%.2f",
            lookup_type,
            source_text,
            query,
            top_match.query_term,
            top_match.score,
            detail,
            accepted_text,
            duration_ms,
        )

    @staticmethod
    def _is_strong_service_match(query_norm: str, match: MatchResult) -> bool:
        record = match.record
        if not isinstance(record, ServiceAreaRecord):
            return False

        candidate_terms = [
            record.canonical_name,
            record.group_name,
            record.road_code,
            record.road_name_core,
            record.road_name_raw,
            *record.preferred_query_terms,
        ]
        for term in candidate_terms:
            normalized_term = _normalize_text(term)
            if not normalized_term:
                continue
            if normalized_term in query_norm or query_norm in normalized_term:
                return True
        return False

    def _match_records(
        self,
        *,
        query: str,
        pool: _RecordPool,
        record_type: RecordType,
        limit: int,
    ) -> list[MatchResult]:
        normalized_query = _normalize_text(query)
        if not normalized_query or not pool.records:
            return []

        candidate_ids = _collect_candidate_ids(normalized_query, pool)
        ranked: list[MatchResult] = []
        for idx in candidate_ids:
            record = pool.records[idx]
            score, reasons = self._score_record(record, normalized_query, record_type=record_type)
            if score <= 0:
                continue
            query_term = record.preferred_query_terms[0] if record.preferred_query_terms else record.group_name
            ranked.append(
                MatchResult(
                    record_type=record_type,
                    score=score,
                    reasons=reasons,
                    query_term=query_term,
                    record=record,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.score,
                len(item.record.group_name),
                item.record.group_name,
                item.record.group_key,
            )
        )
        return ranked[:limit]

    def _score_record(
        self,
        record: ServiceAreaRecord | TollStationRecord,
        query_norm: str,
        *,
        record_type: RecordType,
    ) -> tuple[int, tuple[str, ...]]:
        score = 0
        reasons: list[str] = []

        def add(points: int, reason: str) -> None:
            nonlocal score
            score += points
            reasons.append(reason)

        if query_norm == _normalize_text(record.canonical_name):
            add(1000, f"exact_name:{record.canonical_name}")
        if query_norm == _normalize_text(record.group_name):
            add(980, f"exact_group:{record.group_name}")

        alias_score, alias_reasons = _score_keyword_hits(
            query_norm,
            tuple(_normalize_text(alias) for alias in record.aliases),
            base_score=950,
        )
        if alias_score:
            score = max(score, alias_score)
            reasons.extend(alias_reasons)

        for term in record.search_terms:
            if not term:
                continue
            if term in query_norm:
                add(850, f"contains_term:{term}")
            elif query_norm in term:
                add(700, f"within_term:{term}")

        if record.road_code:
            road_code_norm = _normalize_text(record.road_code)
            if road_code_norm and road_code_norm in query_norm:
                add(400, f"road_code:{record.road_code}")
        if record.road_name_core:
            road_name_core_norm = _normalize_text(record.road_name_core)
            if road_name_core_norm and road_name_core_norm in query_norm:
                add(300, f"road_name_core:{record.road_name_core}")
        if getattr(record, "road_name_raw", ""):
            road_name_raw_norm = _normalize_text(record.road_name_raw)
            if road_name_raw_norm and road_name_raw_norm in query_norm:
                add(250, f"road_name_raw:{record.road_name_raw}")

        if record_type == "service_area" and isinstance(record, ServiceAreaRecord):
            for field_name, value in (
                ("zone", record.geographical_zone),
                ("direction_type", record.direction_type),
                ("direction", record.direction),
                ("direction_name", record.direction_name),
                ("direction_aliasname", record.direction_aliasname),
                ("start_site", record.start_site),
                ("end_site", record.end_site),
            ):
                normalized_value = _normalize_text(value)
                if normalized_value and normalized_value in query_norm:
                    add(200, f"{field_name}:{value}")
        if record_type == "toll_station" and isinstance(record, TollStationRecord):
            normalized_kind = _normalize_text(record.station_kind)
            if normalized_kind and normalized_kind in query_norm:
                add(150, f"station_kind:{record.station_kind}")

        if score == 0:
            fallback_terms = tuple(_normalize_text(term) for term in record.search_terms[:5])
            for term in fallback_terms:
                if term and any(token in query_norm for token in term.split() if token):
                    add(100, f"fallback:{term}")

        return score, tuple(_unique_strings(reasons))


@lru_cache(maxsize=1)
def load_default_facility_catalog() -> FacilityCatalog:
    """Load the generated catalog if it exists."""

    return FacilityCatalog.load_default()
