#!/usr/bin/env python3
"""RouteLLM baseline on the same RouterArena end-to-end protocol.

The dataset loading, prompt construction, scoring, split alignment, and output
schema are shared with ``routerarena_e2e_benchmark.py``.  RouteLLM routing is
performed in-process so the per-sample routing decision and latency are
available even though the upstream RouteLLM server does not expose them.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import routerarena_e2e_benchmark as base


DEFAULT_CONFIG = Path(__file__).with_name("routellm_baseline.yaml")
DEFAULT_OUTPUT_DIR = base.REPO_ROOT / "reports" / "routerarena-e2e" / "routellm"
DEFAULT_BACKEND_URL = "http://10.156.186.8:18081/v1"
DEFAULT_ROUTER = "mf"
DEFAULT_THRESHOLD = 0.11593
DEFAULT_ROUTE_STRONG_MODEL = "gpt-4-1106-preview"
DEFAULT_ROUTE_WEAK_MODEL = "mixtral-8x7b-instruct-v0.1"
DEFAULT_STRONG_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_WEAK_MODEL = "Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RouteLLM baseline using the RouterArena e2e protocol"
    )
    parser.add_argument("--dataset", default=base.DEFAULT_DATASET)
    parser.add_argument(
        "--splits", nargs="+", default=["full"], choices=sorted(base.SUPPORTED_SPLITS)
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--auth-token", default=base.DEFAULT_AUTH_TOKEN)
    parser.add_argument("--user-id", default=base.DEFAULT_USER_ID)
    parser.add_argument("--user-groups", default=base.DEFAULT_USER_GROUPS)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--backend-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--router", default=DEFAULT_ROUTER)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the router-specific routing_threshold from --config",
    )
    parser.add_argument("--route-strong-model", default=DEFAULT_ROUTE_STRONG_MODEL)
    parser.add_argument("--route-weak-model", default=DEFAULT_ROUTE_WEAK_MODEL)
    parser.add_argument("--strong-model", default=DEFAULT_STRONG_MODEL)
    parser.add_argument("--weak-model", default=DEFAULT_WEAK_MODEL)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Write partial detail and summary files after this many samples; use 0 to disable.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=10,
        help="Stop after this many consecutive failed backend requests; use 0 to disable.",
    )
    return parser.parse_args()


def load_routellm(args: argparse.Namespace):
    """Configure RouteLLM's embedding client before importing its routers."""
    if args.embedding_base_url:
        os.environ["OPENAI_BASE_URL"] = args.embedding_base_url
    if args.embedding_api_key:
        os.environ["OPENAI_API_KEY"] = args.embedding_api_key

    import yaml
    from routellm.controller import Controller

    with Path(args.config).expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if args.threshold is None:
        thresholds = config.get("thresholds", {})
        args.threshold = float(thresholds.get(args.router, DEFAULT_THRESHOLD))
    controller = Controller(
        routers=[args.router],
        strong_model=args.route_strong_model,
        weak_model=args.route_weak_model,
        config=config,
        progress_bar=False,
    )
    return controller


def call_backend(
    args: argparse.Namespace, model: str, prompt: str
) -> Tuple[requests.Response, float]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.backend_api_key or args.auth_token}",
        "x-authz-user-id": args.user_id,
        "x-authz-user-groups": args.user_groups,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{args.backend_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=args.timeout,
    )
    return response, (time.perf_counter() - started) * 1000.0


def evaluate_one(
    index: int,
    row: Dict[str, Any],
    args: argparse.Namespace,
    controller: Any,
) -> Dict[str, Any]:
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
        "router": args.router,
        "threshold": args.threshold,
        "route_strong_model": args.route_strong_model,
        "route_weak_model": args.route_weak_model,
    }

    started = time.perf_counter()
    try:
        route_started = time.perf_counter()
        router_score = controller.routers[args.router].calculate_strong_win_rate(prompt)
        routing_latency_ms = (time.perf_counter() - route_started) * 1000.0
        route_decision = (
            "strong" if router_score >= args.threshold else "weak"
        )
        selected_model = args.strong_model if route_decision == "strong" else args.weak_model

        response, backend_elapsed_ms = call_backend(args, selected_model, prompt)
        try:
            payload = response.json()
        except Exception:
            payload = {}
        response_text = base.extract_response_text(payload)
        task_score: Optional[float] = None
        if response.status_code == 200:
            metric_func = base.METRIC_FUNCS.get(metric_name)
            if metric_func is not None:
                task_score = metric_func(response_text, ground_truth, options=row.get("Options"))

        result.update(
            {
                "status": "ok" if response.status_code == 200 else "error",
                "http_status": response.status_code,
                "selected_model": selected_model,
                "route_decision": route_decision,
                "router_score": float(router_score),
                "routing_latency_ms": routing_latency_ms,
                "response_text": response_text,
                "task_score": task_score,
                "is_supported": task_score is not None,
                "http_elapsed_ms": backend_elapsed_ms,
                "actual_http_elapsed_ms": backend_elapsed_ms,
                "raw_response": payload,
            }
        )
        if response.status_code != 200:
            result["error"] = response.text[:2000]
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "http_status": None,
                "error": f"{type(exc).__name__}: {exc}",
                "route_decision": None,
                "router_score": None,
                "routing_latency_ms": None,
                "task_score": None,
                "is_supported": False,
            }
        )
    result["sample_elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def write_split_checkpoint(
    output_dir: Path,
    split: str,
    run_id: str,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    partial: bool,
) -> Dict[str, Any]:
    suffix = "_partial" if partial else ""
    detail_file = output_dir / f"routellm_e2e_{split}_detail_{run_id}{suffix}.jsonl"
    latest_detail_file = output_dir / f"latest_{split}_detail.jsonl"
    base.write_jsonl(detail_file, rows)
    base.write_jsonl(latest_detail_file, rows)
    summary = base.summarize_rows(rows)
    summary.update({
        "run_id": run_id,
        "split": split,
        "detail_file": str(detail_file),
        "router": args.router,
        "threshold": args.threshold,
        "strong_model": args.strong_model,
        "weak_model": args.weak_model,
        "is_partial": partial,
    })
    return summary


def main() -> None:
    args = parse_args()
    controller = load_routellm(args)
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0.0 and 1.0")
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (base.REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_samples = base.load_aligned_splits(args.dataset, args.splits, args.max_samples)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split in args.splits:
        rows: List[Dict[str, Any]] = []
        consecutive_errors = 0
        try:
            for index, sample in enumerate(split_samples.get(split, []), start=1):
                result = evaluate_one(index, sample, args, controller)
                rows.append(result)
                consecutive_errors = consecutive_errors + 1 if result["status"] == "error" else 0
                if args.checkpoint_interval and index % args.checkpoint_interval == 0:
                    write_split_checkpoint(output_dir, split, run_id, rows, args, partial=True)
                    print(f"[checkpoint] {split}: {index}/{len(split_samples[split])}")
                if args.max_consecutive_errors and consecutive_errors >= args.max_consecutive_errors:
                    print(f"[stop] {split}: {consecutive_errors} consecutive errors")
                    break
        except KeyboardInterrupt:
            print(f"[interrupt] Saving {len(rows)} completed {split} samples")
        summary = write_split_checkpoint(
            output_dir,
            split,
            run_id,
            rows,
            args,
            partial=len(rows) < len(split_samples.get(split, [])),
        )
        split_rows[split] = rows
        split_summaries[split] = summary

    robustness = {}
    if "full" in split_rows and "robustness" in split_rows:
        robustness = base.compute_robustness(split_rows["full"], split_rows["robustness"])
    combined = {
        "run_id": run_id,
        "baseline": "RouteLLM",
        "dataset": args.dataset,
        "splits": split_summaries,
        "robustness": robustness,
        "router": args.router,
        "threshold": args.threshold,
        "route_model_pair": {"strong": args.route_strong_model, "weak": args.route_weak_model},
        "backend_model_pair": {"strong": args.strong_model, "weak": args.weak_model},
        "backend_url": args.backend_url,
        "note": "RouteLLM baseline using the semantic-router RouterArena dataset, prompts, metrics, split alignment, detail, and summary protocol.",
    }
    summary_file = output_dir / f"routellm_e2e_summary_{run_id}.json"
    latest_summary = output_dir / "latest_summary.json"
    serialized = json.dumps(combined, indent=2, ensure_ascii=False, default=base.json_default)
    summary_file.write_text(serialized, encoding="utf-8")
    latest_summary.write_text(serialized, encoding="utf-8")
    print("[info] RouteLLM baseline finished")
    print(serialized)


if __name__ == "__main__":
    main()