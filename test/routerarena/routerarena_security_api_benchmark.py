#!/usr/bin/env python3
"""Benchmark the security API with prompts constructed from RouterArena splits.

Unlike routerarena_e2e_benchmark.py, this script calls only
/api/v1/classify/security. It therefore measures the isolated jailbreak
cascade under the same RouterArena prompt distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from routerarena_e2e_benchmark import (
    DEFAULT_DATASET,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_SPLITS,
    REPO_ROOT,
    SUPPORTED_SPLITS,
    build_sample_prompt,
    json_default,
    load_aligned_splits,
)


DEFAULT_ENDPOINT = "/api/v1/classify/security"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "routerarena-e2e" / "security-api"
MAX_CONSECUTIVE_FAILURES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RouterArena prompts through /api/v1/classify/security"
    )
    parser.add_argument(
        "--router-url",
        default="http://localhost:9099",
        help="Classification API base URL",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Security classification endpoint path",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=sorted(SUPPORTED_SPLITS),
        help="RouterArena splits to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples per split. 0 means no limit.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ROUTER_MODEL,
        help="Gateway-required model field",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where detail and summary files are written",
    )
    return parser.parse_args()


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def call_security_api(
    router_url: str, endpoint: str, model: str, prompt: str, timeout: int
) -> tuple[requests.Response, float]:
    started = time.perf_counter()
    response = requests.post(
        f"{router_url.rstrip('/')}{endpoint}",
        json={"model": model, "text": prompt},
        timeout=timeout,
    )
    return response, (time.perf_counter() - started) * 1000.0


def evaluate_one(
    index: int,
    sample: Dict[str, Any],
    router_url: str,
    endpoint: str,
    model: str,
    timeout: int,
) -> Dict[str, Any]:
    dataset_name, metric_name, prompt = build_sample_prompt(sample)
    global_index = sample.get("Global Index") or sample.get("global_index") or sample.get("global index")
    result: Dict[str, Any] = {
        "index": index,
        "global_index": global_index,
        "dataset_name": dataset_name,
        "metric_name": metric_name,
        "question": str(sample.get("Question", "")).strip(),
        "context": str(sample.get("Context", "")).strip(),
        "options": sample.get("Options"),
        "prompt": prompt,
    }

    try:
        response, http_elapsed_ms = call_security_api(
            router_url, endpoint, model, prompt, timeout
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        result.update(
            {
                "status": "ok" if response.status_code == 200 else "error",
                "http_status": response.status_code,
                "http_elapsed_ms": http_elapsed_ms,
                "processing_time_ms": payload.get("processing_time_ms") if isinstance(payload, dict) else None,
                "predicted_is_jailbreak": payload.get("is_jailbreak") if isinstance(payload, dict) else None,
                "confidence": payload.get("confidence") if isinstance(payload, dict) else None,
                "detection_types": payload.get("detection_types", []) if isinstance(payload, dict) else [],
                "recommendation": payload.get("recommendation") if isinstance(payload, dict) else None,
                "raw_response": payload,
            }
        )
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        result.update(
            {
                "status": "error",
                "http_status": None,
                "http_elapsed_ms": None,
                "processing_time_ms": None,
                "predicted_is_jailbreak": None,
                "confidence": None,
                "detection_types": [],
                "recommendation": None,
                "raw_response": None,
                "error": str(exc),
            }
        )
    return result


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    processing_latencies = [
        float(row["processing_time_ms"])
        for row in ok_rows
        if isinstance(row.get("processing_time_ms"), (int, float))
    ]
    http_latencies = [
        float(row["http_elapsed_ms"])
        for row in ok_rows
        if isinstance(row.get("http_elapsed_ms"), (int, float))
    ]
    source_counter = Counter(
        str(row["detection_types"][0])
        for row in ok_rows
        if isinstance(row.get("detection_types"), list) and row["detection_types"]
    )

    return {
        "total_samples": len(rows),
        "ok_count": len(ok_rows),
        "fail_count": len(rows) - len(ok_rows),
        "jailbreak_count": sum(bool(row.get("predicted_is_jailbreak")) for row in ok_rows),
        "avg_processing_time_ms": sum(processing_latencies) / len(processing_latencies) if processing_latencies else None,
        "p50_processing_time_ms": percentile(processing_latencies, 50),
        "p95_processing_time_ms": percentile(processing_latencies, 95),
        "p99_processing_time_ms": percentile(processing_latencies, 99),
        "avg_http_elapsed_ms": sum(http_latencies) / len(http_latencies) if http_latencies else None,
        "p50_http_elapsed_ms": percentile(http_latencies, 50),
        "p95_http_elapsed_ms": percentile(http_latencies, 95),
        "p99_http_elapsed_ms": percentile(http_latencies, 99),
        "detection_type_distribution": dict(source_counter),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    split_samples = load_aligned_splits(args.dataset, args.splits, args.max_samples)
    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split in args.splits:
        samples = split_samples.get(split, [])
        print(f"[info] split={split}, samples={len(samples)}")
        started = time.time()
        rows: List[Dict[str, Any]] = []
        consecutive_failures = 0

        for index, sample in enumerate(samples, start=1):
            row = evaluate_one(
                index, sample, args.router_url, args.endpoint, args.model, args.timeout
            )
            rows.append(row)
            consecutive_failures = 0 if row["status"] == "ok" else consecutive_failures + 1

            if index % 100 == 0:
                print(f"[progress] {split}: {index}/{len(samples)}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[error] {split}: stopped after {MAX_CONSECUTIVE_FAILURES} consecutive failures")
                break

        detail_file = output_dir / f"routerarena_security_api_{split}_detail_{run_id}.jsonl"
        latest_detail_file = output_dir / f"latest_{split}_detail.jsonl"
        write_jsonl(detail_file, rows)
        write_jsonl(latest_detail_file, rows)

        summary = summarize_rows(rows)
        summary.update(
            {
                "run_id": run_id,
                "router_url": args.router_url,
                "endpoint": args.endpoint,
                "dataset": args.dataset,
                "split": split,
                "detail_file": str(detail_file),
                "elapsed_seconds": time.time() - started,
            }
        )
        split_summaries[split] = summary

    combined_summary = {
        "run_id": run_id,
        "router_url": args.router_url,
        "endpoint": args.endpoint,
        "dataset": args.dataset,
        "model": args.model,
        "splits": split_summaries,
        "note": "RouterArena prompts sent only to /api/v1/classify/security. processing_time_ms excludes normal-route signal fan-out.",
    }
    summary_file = output_dir / f"routerarena_security_api_summary_{run_id}.json"
    latest_summary = output_dir / "latest_summary.json"
    rendered = json.dumps(combined_summary, indent=2, ensure_ascii=False, default=json_default)
    summary_file.write_text(rendered, encoding="utf-8")
    latest_summary.write_text(rendered, encoding="utf-8")
    print("[info] benchmark finished")
    print(rendered)


if __name__ == "__main__":
    main()