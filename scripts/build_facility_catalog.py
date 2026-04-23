"""Build the normalized facility catalog JSON from raw Excel sources."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from app.agent.facility_catalog import DEFAULT_CATALOG_PATH, FacilityCatalog

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - 64)
    return index


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        xml = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    shared_strings: list[str] = []
    for si in root.findall("main:si", NS):
        texts = [text.text or "" for text in si.findall(".//main:t", NS)]
        shared_strings.append("".join(texts))
    return shared_strings


def _read_sheet_rows(zf: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[object | None]]:
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[object | None]] = []
    for row in root.findall(".//main:sheetData/main:row", NS):
        values: dict[int, object | None] = {}
        for cell in row.findall("main:c", NS):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            column_index = _col_to_idx(ref)
            cell_type = cell.attrib.get("t")
            value_node = cell.find("main:v", NS)
            value: object | None = None
            if cell_type == "inlineStr":
                text_node = cell.find(".//main:t", NS)
                value = text_node.text if text_node is not None else ""
            elif value_node is not None:
                raw_value = value_node.text
                if cell_type == "s" and raw_value is not None and raw_value.isdigit():
                    value = shared_strings[int(raw_value)]
                else:
                    value = raw_value
            values[column_index] = value
        if values:
            max_index = max(values)
            rows.append([values.get(idx) for idx in range(1, max_index + 1)])
    return rows


def _read_first_sheet(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = _load_shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet = workbook.find("main:sheets/main:sheet", NS)
        if sheet is None:
            raise ValueError(f"No worksheet found in {path}")
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_map[rel_id]
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        rows = _read_sheet_rows(zf, target, shared_strings)

    if not rows:
        return [], []

    header = [str(cell or "").strip() for cell in rows[0]]
    records: list[dict[str, object]] = []
    for row in rows[1:]:
        record: dict[str, object] = {}
        for index, key in enumerate(header):
            if not key:
                continue
            if index < len(row):
                value = row[index]
                if value is None:
                    continue
                record[key] = value
        if record:
            records.append(record)
    return header, records


def _default_input_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "tmp" / "app_service_info.xlsx",
        root / "tmp" / "app_toll_info.xlsx",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-xlsx", type=Path, default=None)
    parser.add_argument("--toll-xlsx", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()

    default_service, default_toll = _default_input_paths()
    service_path = args.service_xlsx or default_service
    toll_path = args.toll_xlsx or default_toll

    _, service_rows = _read_first_sheet(service_path)
    _, toll_rows = _read_first_sheet(toll_path)

    catalog = FacilityCatalog.from_raw_rows(service_rows=service_rows, toll_rows=toll_rows)
    output_path = catalog.save_json(args.output)

    print(
        json.dumps(
            {
                "service_rows": len(service_rows),
                "toll_rows": len(toll_rows),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
