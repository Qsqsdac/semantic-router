#!/usr/bin/env python3
"""M-IRT baseline using the shared RouterArena end-to-end protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

import routerarena_e2e_benchmark as base


DEFAULT_OUTPUT_DIR = base.REPO_ROOT / "reports" / "routerarena-e2e" / "mirt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M-IRT baseline on RouterArena")
    parser.add_argument("--router-url", default="http://localhost:8088")
    parser.add_argument("--endpoint", default=base.DEFAULT_ENDPOINT)
    parser.add_argument("--dataset", default=base.DEFAULT_DATASET)
    parser.add_argument("--splits", nargs="+", default=["full"], choices=sorted(base.SUPPORTED_SPLITS))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--auth-token", default=base.DEFAULT_AUTH_TOKEN)
    parser.add_argument("--user-id", default=base.DEFAULT_USER_ID)
    parser.add_argument("--user-groups", default=base.DEFAULT_USER_GROUPS)
    parser.add_argument("--model", default="mirt-router")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    return parser.parse_args()


def evaluate_one(index: int, row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    dataset_name, metric_name, prompt = base.build_sample_prompt(row)
    ground_truth = row.get("Answer", "")
    result: Dict[str, Any] = {
        "index": index,
        "global_index": row.get("Global Index") or row.get("global_index") or row.get("global index"),
        "dataset_name": dataset_name,
        "metric_name": metric_name,
        "expected_raw": ground_truth,
        "question": str(row.get("Question", "")).strip(),
        "context": str(row.get("Context", "")).strip(),
        "options": row.get("Options"),
        "prompt": prompt,
        "router": "mirt",
        "candidate_set_version": "mirt-bert-zero-shot-v1",
    }
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    headers = {
        "Authorization": f"Bearer {args.auth_token}",
        "x-authz-user-id": args.user_id,
        "x-authz-user-groups": args.user_groups,
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{args.router_url.rstrip('/')}{args.endpoint}",
            headers=headers,
            json=payload,
            timeout=args.timeout,
        )
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {}
        response_text = base.extract_response_text(response_payload)
        task_score = None
        if response.status_code == 200:
            metric_func = base.METRIC_FUNCS.get(metric_name)
            if metric_func is not None:
                task_score = metric_func(response_text, ground_truth, options=row.get("Options"))
        result.update(
            {
                "status": "ok" if response.status_code == 200 else "error",
                "http_status": response.status_code,
                "selected_model": response.headers.get("x-mirt-selected-model") or response_payload.get("model"),
                "response_text": response_text,
                "task_score": task_score,
                "is_supported": task_score is not None,
                "raw_response": response_payload,
                "routing_latency_ms": response.headers.get("x-mirt-routing-latency-ms"),
                "predicted_success": response.headers.get("x-mirt-predicted-success"),
                "utility": response.headers.get("x-mirt-utility"),
                "estimated_cost": response.headers.get("x-mirt-estimated-cost"),
                "actual_cost": response.headers.get("x-mirt-actual-cost"),
                "http_elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        )
        if response.status_code != 200:
            result["error"] = response.text[:2000]
    except Exception as exc:
        result.update({"status": "error", "http_status": None, "error": f"{type(exc).__name__}: {exc}"})
    result["sample_elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


def write_results(output_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace, run_id: str) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_file = output_dir / f"mirt_e2e_full_detail_{run_id}.jsonl"
    base.write_jsonl(detail_file, rows)
    summary = base.summarize_rows(rows)
    summary.update({"run_id": run_id, "baseline": "M-IRT", "detail_file": str(detail_file), "router_url": args.router_url})
    serialized = json.dumps(summary, indent=2, ensure_ascii=False, default=base.json_default)
    (output_dir / f"mirt_e2e_summary_{run_id}.json").write_text(serialized, encoding="utf-8")
    (output_dir / "latest_summary.json").write_text(serialized, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    samples = base.load_aligned_splits(args.dataset, args.splits, args.max_samples)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    for split in args.splits:
        rows = [evaluate_one(index, row, args) for index, row in enumerate(samples[split], start=1)]
        summary = write_results(output_dir / split, rows, args, run_id)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=base.json_default))


if __name__ == "__main__":
    main()