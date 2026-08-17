#!/usr/bin/env python3
"""Benchmark semantic-cache hits using the same MMLU-Pro samples as intent eval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from eval_classification_api_intent import (
    DEFAULT_DATASET,
    REPO_ROOT,
    extract_eval_fields,
    load_hf_dataset,
    print_dataset_stats,
    resolve_dataset_split,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "semantic-cache-mock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic-cache benchmark through /v1/chat/completions"
    )
    parser.add_argument("--router-url", default="http://localhost:9099")
    parser.add_argument("--mock-url", default="http://localhost:18081")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--model", default="MoM")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--reset-mock-stats",
        action="store_true",
        help="Reset mock backend counters before running.",
    )
    return parser.parse_args()


def post_chat(router_url: str, model: str, text: str, timeout: int) -> tuple[requests.Response, float]:
    url = f"{router_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 16,
    }
    start = time.time()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.time() - start) * 1000
    return response, elapsed_ms


def get_json(url: str, timeout: int) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def reset_mock(mock_url: str, timeout: int) -> None:
    response = requests.post(f"{mock_url.rstrip()}/stats/reset", json={}, timeout=timeout)
    response.raise_for_status()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    split = resolve_dataset_split(args.dataset, args.split)

    health = requests.get(f"{args.mock_url.rstrip('/')}/healthz", timeout=args.timeout)
    health.raise_for_status()
    router_health = requests.get(f"{args.router_url.rstrip('/')}/health", timeout=args.timeout)
    if router_health.status_code >= 500:
        router_health.raise_for_status()

    if args.reset_mock_stats:
        reset_mock(args.mock_url, args.timeout)

    samples = load_hf_dataset(args.dataset, split, args.max_samples)
    print(f"[info] loaded samples: {len(samples)}")
    print_dataset_stats(args.dataset, split, samples)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()

    details: list[dict[str, Any]] = []
    pass_summaries: list[dict[str, Any]] = []
    total_started = time.time()

    for pass_index in range(1, args.passes + 1):
        started = time.time()
        ok_count = 0
        fail_count = 0
        cache_hits = 0
        cache_misses = 0
        status_counts: dict[str, int] = {}
        latencies: list[float] = []

        for idx, sample in enumerate(samples, start=1):
            fields = extract_eval_fields(sample, args.dataset)
            text = fields["text"]
            row: dict[str, Any] = {
                "run_id": run_id,
                "pass": pass_index,
                "index": idx,
                "dataset": args.dataset,
                "split": split,
                "model": args.model,
                "question": text,
                "expected_raw": fields["expected_raw"],
                "expected_norm": fields["expected_norm"],
            }
            try:
                response, elapsed_ms = post_chat(args.router_url, args.model, text, args.timeout)
                cache_hit = response.headers.get("x-vsr-cache-hit", "").lower() == "true"
                status_counts[str(response.status_code)] = status_counts.get(str(response.status_code), 0) + 1
                response.raise_for_status()
                body = response.json()
                ok_count += 1
                if cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
                latencies.append(elapsed_ms)
                row.update(
                    {
                        "status": "ok",
                        "http_status": response.status_code,
                        "cache_hit": cache_hit,
                        "elapsed_ms": elapsed_ms,
                        "selected_decision": response.headers.get("x-vsr-selected-decision"),
                        "selected_category": response.headers.get("x-vsr-selected-category"),
                        "selected_model": response.headers.get("x-vsr-selected-model"),
                        "response_id": body.get("id"),
                        "response_model": body.get("model"),
                        "content": (
                            body.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content")
                        ),
                    }
                )
            except Exception as exc:
                fail_count += 1
                row.update({"status": "error", "error": str(exc)})

            details.append(row)
            if idx % 100 == 0:
                print(f"[progress] pass {pass_index}/{args.passes}: {idx}/{len(samples)}")

        elapsed = time.time() - started
        hit_rate = cache_hits / ok_count if ok_count else 0.0
        avg_latency_ms = sum(latencies) / len(latencies) if latencies else None
        pass_summary = {
            "pass": pass_index,
            "total_samples": len(samples),
            "ok_count": ok_count,
            "fail_count": fail_count,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": hit_rate,
            "avg_latency_ms": avg_latency_ms,
            "elapsed_seconds": elapsed,
            "http_status_counts": status_counts,
        }
        pass_summaries.append(pass_summary)
        print(
            f"[pass {pass_index}] ok={ok_count}, errors={fail_count}, "
            f"hits={cache_hits}, misses={cache_misses}, hit_rate={hit_rate:.4f}, "
            f"avg_latency_ms={(avg_latency_ms or 0):.2f}"
        )

    mock_stats = get_json(f"{args.mock_url.rstrip('/')}/stats", args.timeout)
    total_elapsed = time.time() - total_started
    detail_file = out_dir / f"semantic_cache_mock_detail_{run_id}.jsonl"
    summary_file = out_dir / f"semantic_cache_mock_summary_{run_id}.json"
    latest_detail = out_dir / "latest_detail.jsonl"
    latest_summary = out_dir / "latest_summary.json"
    write_jsonl(detail_file, details)
    write_jsonl(latest_detail, details)

    summary = {
        "run_id": run_id,
        "router_url": args.router_url,
        "mock_url": args.mock_url,
        "endpoint": "/v1/chat/completions",
        "dataset": args.dataset,
        "split": split,
        "model": args.model,
        "passes": args.passes,
        "total_elapsed_seconds": total_elapsed,
        "pass_summaries": pass_summaries,
        "mock_backend_stats": mock_stats,
        "detail_file": str(detail_file),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Semantic Cache Benchmark Done ===")
    print(f"run_id: {run_id}")
    for item in pass_summaries:
        print(
            f"pass {item['pass']}: hit_rate={item['cache_hit_rate']:.4f} "
            f"({item['cache_hits']}/{item['ok_count']}), "
            f"avg_latency_ms={(item['avg_latency_ms'] or 0):.2f}, "
            f"errors={item['fail_count']}"
        )
    print(f"mock backend chat_completions: {mock_stats.get('chat_completions')}")
    print(f"detail: {detail_file}")
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
