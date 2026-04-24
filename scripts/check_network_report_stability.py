"""Probe the network-report answer for formatting stability across repeated calls.

The script is intentionally small and self-contained so it can be used both as a
quick manual checker and as a repeatable regression probe.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3535ba3b"
DEFAULT_ITERATIONS = 20

DEFAULT_PROMPT = (
    "请提供省内整体实时路况总结。\n"
    "请先输出 1-2 句中文播报总结，再输出 Markdown 表格。\n"
    "表头固定为：| roadCode | highwayName | roadSection | controls | traffic |\n"
    "不要输出“序号”等额外列，不要改写表头字段名，不要在表格前重复解释字段含义。\n"
    "controls 和 traffic 多个值请用、分隔，没有请填无"
)

EXPECTED_TABLE_HEADER = "| roadCode | highwayName | roadSection | controls | traffic |"
FORBIDDEN_PHRASES = [
    "AI播报总结",
    "请先输出",
    "不要输出",
    "不要改写",
    "不要在表格前重复解释字段含义",
    "表头固定为",
    "controls 和 traffic",
    "有遵从用户的问题中的内容",
]


@dataclass(slots=True)
class ProbeResult:
    index: int
    ok: bool
    latency_ms: float
    summary: str
    table_lines: list[str]
    raw_text: str
    violations: list[str]


def _build_payload(prompt: str, *, model: str, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "enable_thinking": False,
    }


def _extract_sse_text(raw_text: str) -> str:
    content_parts: list[str] = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue

        data = block.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue

        payload = json.loads(data)
        if "error" in payload:
            raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))

        for choice in payload.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)

    return "".join(content_parts)


def _split_summary_and_table(text: str) -> tuple[str, list[str]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    table_start = next((idx for idx, line in enumerate(lines) if line.startswith("|")), len(lines))
    summary = "\n".join(lines[:table_start]).strip()
    table_lines = lines[table_start:]
    return summary, table_lines


def _validate_output(text: str) -> list[str]:
    violations: list[str] = []

    if EXPECTED_TABLE_HEADER not in text:
        violations.append("missing_expected_table_header")
    if "| 序号 |" in text or "序号" in text:
        violations.append("unexpected_serial_number_column")

    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            violations.append(f"instruction_leak:{phrase}")

    summary, table_lines = _split_summary_and_table(text)
    if not summary:
        violations.append("missing_summary")
    if not table_lines:
        violations.append("missing_table")

    for line in table_lines:
        if not line.startswith("|"):
            continue
        column_count = len([cell for cell in line.split("|") if cell.strip()])
        if column_count != 5:
            violations.append(f"bad_column_count:{column_count}:{line}")
            break

    return violations


def _probe_once(
    client: httpx.Client,
    *,
    url: str,
    payload: dict[str, Any],
    stream: bool,
    index: int,
) -> ProbeResult:
    start = time.perf_counter()
    if stream:
        with client.stream("POST", url, json=payload) as response:
            raw_bytes = response.read()
            status_code = response.status_code
    else:
        response = client.post(url, json=payload)
        raw_bytes = response.content
        status_code = response.status_code

    latency_ms = (time.perf_counter() - start) * 1000.0
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    if status_code != 200:
        return ProbeResult(
            index=index,
            ok=False,
            latency_ms=latency_ms,
            summary="",
            table_lines=[],
            raw_text=raw_text,
            violations=[f"unexpected_status:{status_code}"],
        )

    output_text = _extract_sse_text(raw_text) if stream else raw_text
    summary, table_lines = _split_summary_and_table(output_text)
    violations = _validate_output(output_text)
    return ProbeResult(
        index=index,
        ok=not violations,
        latency_ms=latency_ms,
        summary=summary,
        table_lines=table_lines,
        raw_text=output_text,
        violations=violations,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--stream", dest="stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    payload = _build_payload(args.prompt, model=args.model, stream=args.stream)
    timeout = httpx.Timeout(
        timeout=args.timeout,
        connect=args.connect_timeout,
        read=args.timeout,
        write=args.timeout,
        pool=args.timeout,
    )

    results: list[ProbeResult] = []
    with httpx.Client(timeout=timeout) as client:
        for index in range(1, args.iterations + 1):
            result = _probe_once(client, url=args.url, payload=payload, stream=args.stream, index=index)
            results.append(result)
            if args.sleep > 0 and index < args.iterations:
                time.sleep(args.sleep)

    failed = [result for result in results if not result.ok]
    latencies = [result.latency_ms for result in results]
    stable_rate = ((len(results) - len(failed)) / len(results) * 100.0) if results else 0.0

    print(f"Endpoint: {args.url}")
    print(f"Model: {args.model}")
    print(f"Iterations: {len(results)}")
    print(f"Stream: {args.stream}")
    print(f"Stable rate: {stable_rate:.1f}%")
    if len(latencies) >= 2:
        p95 = statistics.quantiles(latencies, n=20)[18]
    else:
        p95 = latencies[0]

    print(
        "Latency ms: "
        f"min={min(latencies):.1f} "
        f"avg={statistics.fmean(latencies):.1f} "
        f"p95={p95:.1f} "
        f"max={max(latencies):.1f}"
    )

    if failed:
        print("\nFirst failing samples:")
        for result in failed[:5]:
            print(f"- #{result.index}: {', '.join(result.violations)}")
            print("  summary:", result.summary[:200].replace("\n", " "))
            print("  table:", " | ".join(result.table_lines[:3]))
        return 1

    print("\nAll probes passed.")
    if results:
        example = results[0]
        print("Summary preview:", example.summary)
        print("Table preview:")
        for line in example.table_lines[:4]:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
