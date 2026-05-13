#!/usr/bin/env python3
"""批量测试“摩托车能否上高速”类问题的回答准确度。

脚本只依赖 Python 标准库，适合在无网络 Linux 环境运行；前提是聊天服务
已经在本机启动，并能通过 /v1/chat/completions 接口访问。

判定标准：
- 只有回答明确表达“摩托车不能 / 不允许 / 禁止进入或通行高速公路”才算正确。
- 如果回答出现“多数省份允许”“可以上高速”等泛化或肯定表述，会判为失败。

用法示例：
- 默认执行 200 次，并生成 motorcycle_highway_access_results.csv：
  python scripts/test_motorcycle_highway_access_accuracy.py
- 执行 1000 次，并发 5 个请求：
  python scripts/test_motorcycle_highway_access_accuracy.py -n 1000 -c 5
- 指定接口、模型和 CSV 输出文件：
  python scripts/test_motorcycle_highway_access_accuracy.py \
    --endpoint http://127.0.0.1:8090/v1/chat/completions \
    --model qwen3535ba3b \
    --csv result.csv
- 输出每一条测试结果，便于排查失败样本：
  python scripts/test_motorcycle_highway_access_accuracy.py -n 200 --full
- 运行过程中按 Ctrl+C 会停止后续测试，并把已完成结果写入 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3535ba3b"
DEFAULT_ITERATIONS = 200
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CONCURRENCY = 1
DEFAULT_CSV_PATH = "motorcycle_highway_access_results.csv"

# QUESTION_VARIANTS = (
#     "摩托车能上吗",
#     "摩托车可以上吗",
#     "摩托车能上高速吗",
#     "摩托车可以上高速吗",
#     "摩托车能上高速公路吗",
#     "摩托车可以走高速吗",
#     "摩托车能不能上高速",
#     "摩托车可不可以上高速",
#     "摩托能上吗",
#     "摩托能上高速吗",
#     "机车能上高速吗",
#     "两轮摩托车能上高速吗",
#     "浙江摩托车能上高速吗",
#     "浙江省摩托车可以上高速公路吗",
#     "在浙江摩托车能不能上高速",
# )
QUESTION_VARIANTS = (
    "摩托车能上吗",
    "摩托车可以上吗",
    "摩托车能上高速吗"
)

DENY_PATTERNS = (
    re.compile(r"(摩托车|摩托|机车|两轮摩托车).{0,12}(不能|不可|不可以|不允许|禁止|不得|不准|禁行).{0,16}(上|进入|驶入|通行|行驶|走).{0,8}(高速|高速公路)?"),
    re.compile(r"(不能|不可|不可以|不允许|禁止|不得|不准|禁行).{0,16}(摩托车|摩托|机车|两轮摩托车).{0,16}(上|进入|驶入|通行|行驶|走).{0,8}(高速|高速公路)?"),
    re.compile(r"(摩托车|摩托|机车|两轮摩托车).{0,12}(禁止进入|禁止驶入|禁止通行|禁止上).{0,8}(高速|高速公路)?"),
    re.compile(r"(高速|高速公路).{0,12}(禁止|不得|不允许).{0,12}(摩托车|摩托|机车|两轮摩托车).{0,12}(进入|驶入|通行|行驶)?"),
)

POSITIVE_PATTERNS = (
    re.compile(r"(?<!不)(?<!非)(?<!禁止)(可以|可|允许|能|能够).{0,8}(上|进入|驶入|通行|行驶|走).{0,8}(高速|高速公路)"),
    re.compile(r"(高速|高速公路).{0,8}(可以|允许).{0,8}(摩托车|摩托|机车|两轮摩托车).{0,8}(进入|通行|行驶)"),
)


@dataclass(frozen=True)
class CaseResult:
    index: int
    question: str
    ok: bool
    answer: str
    reason: str
    elapsed_ms: int
    error: str | None = None


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def is_correct_answer(answer: str) -> tuple[bool, str]:
    """Return True only when the answer clearly says motorcycles are not allowed."""

    normalized = normalize_text(answer)
    if not normalized:
        return False, "empty_answer"

    has_deny = any(pattern.search(normalized) for pattern in DENY_PATTERNS)
    if not has_deny:
        return False, "missing_clear_denial"

    positive_hits = [
        match.group(0)
        for pattern in POSITIVE_PATTERNS
        for match in [pattern.search(normalized)]
        if match is not None
    ]
    if positive_hits and "浙江" not in normalized:
        return False, "contains_positive_highway_access"

    if any(phrase in normalized for phrase in ("全国多数省份允许", "多数省份允许", "可以上高速公路通行")):
        return False, "contains_general_positive_claim"

    return True, "clear_denial"


def build_payload(question: str, model: str, brief_answer: bool, enable_thinking: bool) -> bytes:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "enable_thinking": enable_thinking,
        "brief_answer": brief_answer,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def extract_answer(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def ask_once(
    *,
    index: int,
    question: str,
    endpoint: str,
    model: str,
    timeout_seconds: float,
    brief_answer: bool,
    enable_thinking: bool,
) -> CaseResult:
    started_at = time.perf_counter()
    request = urllib.request.Request(
        endpoint,
        data=build_payload(question, model, brief_answer, enable_thinking),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        answer = extract_answer(payload if isinstance(payload, dict) else {})
        ok, reason = is_correct_answer(answer)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return CaseResult(index, question, ok, answer, reason, elapsed_ms)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return CaseResult(index, question, False, body, "http_error", elapsed_ms, str(exc))
    except Exception as exc:  # noqa: BLE001 - standalone diagnostic script
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return CaseResult(index, question, False, "", "request_error", elapsed_ms, repr(exc))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量测试“摩托车能否上高速”类问题回答是否明确判断为不允许。",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"接口地址，默认 {DEFAULT_ENDPOINT}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"请求 model 字段，默认 {DEFAULT_MODEL}")
    parser.add_argument("-n", "--iterations", type=int, default=DEFAULT_ITERATIONS, help="测试次数，建议 200-1000")
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发数，默认 1")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="单次请求超时时间，秒")
    parser.add_argument("--seed", type=int, default=20260513, help="随机种子，便于复现")
    parser.add_argument("--full", action="store_true", help="输出每一条测试结果")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help=f"CSV 明细输出路径，默认 {DEFAULT_CSV_PATH}")
    parser.add_argument("--jsonl", default="", help="可选：把每条结果写入 JSONL 文件")
    parser.add_argument("--brief-answer", action="store_true", default=True, help="使用 brief_answer=true，默认开启")
    parser.add_argument("--no-brief-answer", dest="brief_answer", action="store_false", help="使用 brief_answer=false")
    parser.add_argument("--enable-thinking", action="store_true", default=False, help="使用 enable_thinking=true，默认 false")
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations 必须大于 0")
    if args.iterations > 1000:
        parser.error("--iterations 最大限制为 1000，避免误压服务")
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于 0")
    return args


def write_jsonl(path: str, results: list[CaseResult]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")


def write_csv(path: str, results: list[CaseResult]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "question",
                "ok",
                "reason",
                "elapsed_ms",
                "error",
                "answer",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "index": result.index,
                    "question": result.question,
                    "ok": "true" if result.ok else "false",
                    "reason": result.reason,
                    "elapsed_ms": result.elapsed_ms,
                    "error": result.error or "",
                    "answer": result.answer,
                }
            )


def print_result(result: CaseResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(
        f"[{status}] #{result.index} {result.elapsed_ms}ms "
        f"reason={result.reason} question={result.question} answer={result.answer}"
    )


def summarize_and_write_outputs(args: argparse.Namespace, results: list[CaseResult], *, interrupted: bool) -> int:
    results.sort(key=lambda item: item.index)
    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    accuracy = passed / len(results) if results else 0.0
    elapsed_values = [result.elapsed_ms for result in results]
    avg_elapsed = int(sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0
    max_elapsed = max(elapsed_values) if elapsed_values else 0

    if args.csv:
        write_csv(args.csv, results)
        print(f"csv: {args.csv}")
    if args.jsonl:
        write_jsonl(args.jsonl, results)
        print(f"jsonl: {args.jsonl}")

    prefix = "interrupted" if interrupted else "summary"
    print(
        f"{prefix}: "
        f"passed={passed} failed={failed} total={len(results)} "
        f"accuracy={accuracy:.2%} avg_ms={avg_elapsed} max_ms={max_elapsed}"
    )
    if interrupted:
        return 130
    return 0 if failed == 0 else 1


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    questions = [random.choice(QUESTION_VARIANTS) for _ in range(args.iterations)]

    print(
        "start: "
        f"endpoint={args.endpoint} iterations={args.iterations} "
        f"concurrency={args.concurrency} brief_answer={args.brief_answer} "
        f"enable_thinking={args.enable_thinking}"
    )

    results: list[CaseResult] = []
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = []
    try:
        futures = [
            executor.submit(
                ask_once,
                index=index,
                question=question,
                endpoint=args.endpoint,
                model=args.model,
                timeout_seconds=args.timeout,
                brief_answer=args.brief_answer,
                enable_thinking=args.enable_thinking,
            )
            for index, question in enumerate(questions, start=1)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if args.full or not result.ok:
                print_result(result)
    except KeyboardInterrupt:
        print("\nreceived Ctrl+C, writing completed results and exiting...")
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        exit_code = summarize_and_write_outputs(args, results, interrupted=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return summarize_and_write_outputs(args, results, interrupted=False)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
