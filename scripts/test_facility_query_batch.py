"""批量测试服务区和收费站是否能通过 chat completions API 查询成功。

脚本会从 app_service_info.xlsx 和 app_toll_info.xlsx 中读取 name 列，
把每个服务区或收费站名称作为用户问题发送到接口，并把每次请求结果写入 CSV 表格。
脚本只使用 Python 标准库，不依赖 pandas/openpyxl，方便在 Linux 测试环境直接运行。

使用示例：

    # 同时测试服务区和收费站。
    python scripts/test_facility_query_batch.py \
      --service-xlsx /data/app_service_info.xlsx \
      --toll-xlsx /data/app_toll_info.xlsx \
      --target all \
      --url http://127.0.0.1:8090/v1/chat/completions \
      --model qwen3535ba3b \
      --concurrency 2 \
      --request-interval 1 \
      --output facility_query_report.csv

    # 只测试服务区。
    python scripts/test_facility_query_batch.py \
      --service-xlsx /tmp/app_service_info.xlsx \
      --target service \
      --concurrency 2 \
      --output service_query_report.csv

    # 只测试收费站。
    python scripts/test_facility_query_batch.py \
      --toll-xlsx /data/app_toll_info.xlsx \
      --target toll \
      --output toll_query_report.csv

常用参数说明：

    --target              测试范围，可选 all、service、toll。
    --concurrency         最大并发请求数，默认 1。
    --request-interval    两次请求开始之间的最小间隔秒数，默认 1。
    --timeout             单个请求超时时间，默认 60 秒。
    --limit               只测试前 N 条数据，默认 0 表示全部测试。
    --unique-name         同一类型下相同名称只测试一次。
    --output              CSV 结果表路径。
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3535ba3b"
NO_CONTENT_KEYWORDS = (
    "未查询到",
    "没有查询到",
    "无法查询",
    "暂无",
    "无相关",
    "没有相关",
    "未找到",
    "找不到",
)


class RequestLimiter:
    """Limit request start interval across worker threads."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return

        with self._lock:
            now = time.monotonic()
            sleep_seconds = self._next_request_at - now
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
                now = time.monotonic()
            self._next_request_at = now + self.min_interval_seconds


def _column_to_index(cell_ref: str) -> int:
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


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text_node = cell.find(".//main:t", NS)
        return text_node.text if text_node is not None else ""

    value_node = cell.find("main:v", NS)
    if value_node is None:
        return None

    raw_value = value_node.text or ""
    if cell_type == "s" and raw_value.isdigit():
        index = int(raw_value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
    return raw_value


def _read_sheet_rows(
    zf: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[list[str | None]]:
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[str | None]] = []

    for row in root.findall(".//main:sheetData/main:row", NS):
        values: dict[int, str | None] = {}
        for cell in row.findall("main:c", NS):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            values[_column_to_index(ref)] = _cell_value(cell, shared_strings)

        if values:
            max_index = max(values)
            rows.append([values.get(idx) for idx in range(1, max_index + 1)])

    return rows


def read_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet from an xlsx file as dictionaries."""

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
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        rows = _read_sheet_rows(zf, sheet_path, shared_strings)

    if not rows:
        return []

    headers = [str(cell or "").strip() for cell in rows[0]]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        record: dict[str, str] = {}
        for index, header in enumerate(headers):
            if not header:
                continue
            if index < len(row) and row[index] is not None:
                record[header] = str(row[index]).strip()
        if record:
            records.append(record)
    return records


def load_cases(
    service_xlsx: Path,
    toll_xlsx: Path,
    target: str,
    unique_name: bool,
) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    if target in {"all", "service"}:
        for row_index, row in enumerate(read_first_sheet(service_xlsx), start=2):
            name = row.get("name", "").strip()
            if name:
                cases.append({"case_type": "service", "source_row": str(row_index), **row})

    if target in {"all", "toll"}:
        for row_index, row in enumerate(read_first_sheet(toll_xlsx), start=2):
            name = row.get("name", "").strip()
            if name:
                cases.append({"case_type": "toll", "source_row": str(row_index), **row})

    if not unique_name:
        return cases

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for case in cases:
        key = (case["case_type"], case["name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def build_payload(
    name: str,
    model: str,
    brief_answer: bool,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": name}],
        "stream": False,
        "enable_thinking": enable_thinking,
        "brief_answer": brief_answer,
    }


def extract_answer(response_json: Any) -> str:
    if not isinstance(response_json, dict):
        return ""

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                return json.dumps(content, ensure_ascii=False)
            text = first_choice.get("text")
            if isinstance(text, str):
                return text
    return ""


def has_related_content(name: str, answer: str) -> bool:
    compact_answer = answer.strip()
    if not compact_answer:
        return False
    if any(keyword in compact_answer for keyword in NO_CONTENT_KEYWORDS):
        return False

    normalized_name = name.replace("（", "(").replace("）", ")")
    normalized_answer = compact_answer.replace("（", "(").replace("）", ")")
    return name in compact_answer or normalized_name in normalized_answer


def post_chat_completion(
    case: dict[str, str],
    url: str,
    model: str,
    timeout: float,
    brief_answer: bool,
    enable_thinking: bool,
    limiter: RequestLimiter,
) -> dict[str, str]:
    name = case["name"]
    payload = build_payload(name, model, brief_answer, enable_thinking)
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = datetime.now().isoformat(timespec="seconds")
    elapsed_ms = 0
    http_status = ""
    answer = ""
    raw_response = ""
    error = ""

    try:
        limiter.wait()
        start_time = time.monotonic()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = str(response.status)
            response_body = response.read().decode("utf-8", errors="replace")
        elapsed_ms = round((time.monotonic() - start_time) * 1000)
        raw_response = response_body
        response_json = json.loads(response_body)
        answer = extract_answer(response_json)
    except urllib.error.HTTPError as exc:
        http_status = str(exc.code)
        elapsed_ms = (
            round((time.monotonic() - start_time) * 1000) if "start_time" in locals() else 0
        )
        raw_response = exc.read().decode("utf-8", errors="replace")
        error = f"HTTPError: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - each failed case must be recorded and continue.
        elapsed_ms = (
            round((time.monotonic() - start_time) * 1000) if "start_time" in locals() else 0
        )
        error = f"{type(exc).__name__}: {exc}"

    request_ok = http_status.startswith("2") and not error
    related = has_related_content(name, answer)
    result = {
        **case,
        "query": name,
        "started_at": started_at,
        "elapsed_ms": str(elapsed_ms),
        "http_status": http_status,
        "request_ok": "Y" if request_ok else "N",
        "has_answer": "Y" if bool(answer.strip()) else "N",
        "name_in_answer": "Y" if name in answer else "N",
        "related_content": "Y" if related else "N",
        "answer": answer,
        "raw_response": raw_response,
        "error": error,
    }
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    base_fields = [
        "case_type",
        "source_row",
        "name",
        "query",
        "started_at",
        "elapsed_ms",
        "http_status",
        "request_ok",
        "has_answer",
        "name_in_answer",
        "related_content",
        "answer",
        "error",
        "raw_response",
    ]
    extra_fields = sorted({key for row in rows for key in row if key not in base_fields})
    fieldnames = base_fields[:3] + extra_fields + base_fields[3:]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="chat completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name sent in request payload")
    parser.add_argument(
        "--target",
        choices=["all", "service", "toll"],
        default="all",
        help="which xlsx source to test",
    )
    parser.add_argument("--service-xlsx", type=Path, default=Path("app_service_info.xlsx"))
    parser.add_argument("--toll-xlsx", type=Path, default=Path("app_toll_info.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"facility_query_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
    )
    parser.add_argument("--concurrency", type=int, default=1, help="max parallel requests")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.0,
        help="minimum seconds between starting two requests",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="request timeout seconds")
    parser.add_argument("--limit", type=int, default=0, help="only test first N cases; 0 means all")
    parser.add_argument(
        "--unique-name",
        action="store_true",
        help="test each name once per target type",
    )
    parser.add_argument(
        "--brief-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send brief_answer in request payload",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="send enable_thinking in request payload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.request_interval < 0:
        raise ValueError("--request-interval must be >= 0")

    cases = load_cases(args.service_xlsx, args.toll_xlsx, args.target, args.unique_name)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No test cases found. Check --target and xlsx paths.")

    print(
        f"Loaded {len(cases)} cases; target={args.target}; "
        f"concurrency={args.concurrency}; interval={args.request_interval}s"
    )

    limiter = RequestLimiter(args.request_interval)
    results: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(
                post_chat_completion,
                case,
                args.url,
                args.model,
                args.timeout,
                args.brief_answer,
                args.enable_thinking,
                limiter,
            ): case
            for case in cases
        }

        for future in as_completed(future_map):
            case = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep batch running even on unexpected errors.
                result = {
                    **case,
                    "query": case.get("name", ""),
                    "request_ok": "N",
                    "has_answer": "N",
                    "name_in_answer": "N",
                    "related_content": "N",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            completed += 1
            status = result.get("request_ok", "N")
            print(
                f"[{completed}/{len(cases)}] "
                f"{case['case_type']} {case['name']} request_ok={status}"
            )

    results.sort(key=lambda row: (row.get("case_type", ""), int(row.get("source_row", "0") or 0)))
    write_csv(args.output, results)

    ok_count = sum(1 for row in results if row.get("request_ok") == "Y")
    related_count = sum(1 for row in results if row.get("related_content") == "Y")
    print(
        f"Done. request_ok={ok_count}/{len(results)}, "
        f"related_content={related_count}/{len(results)}, output={args.output}"
    )


if __name__ == "__main__":
    main()
