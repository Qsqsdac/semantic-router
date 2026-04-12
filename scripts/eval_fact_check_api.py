#!/usr/bin/env python3
"""
面向 Classification API 的 fact-check 规模化评测脚本。

说明：
1) 默认使用 llm-semantic-router/fact-check-classification-dataset。
2) 目标接口是 /api/v1/classify/fact-check。
3) 保存逐条样本输出和汇总指标，便于回归比较。

示例：
python scripts/eval_fact_check_api.py \
  --router-url http://localhost:8080 \
  --max-samples 1000 \
  --output-dir reports/classification-fact-check
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_DATASET = "llm-semantic-router/fact-check-classification-dataset"
SUPPORTED_DATASETS: Dict[str, Dict[str, str]] = {
    DEFAULT_DATASET: {
        "default_split": "test",
        "text_field": "text",
        "label_field": "label_id",
    },
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "classification-fact-check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-driven benchmark for /api/v1/classify/fact-check"
    )
    parser.add_argument(
        "--router-url",
        default="http://localhost:8080",
        help="Classification API base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HF dataset for fact-check benchmark (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split, if omitted uses dataset default split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max sample count, 0 means all",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent request workers for batch performance testing",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save detail/summary files",
    )
    return parser.parse_args()


def resolve_dataset_split(dataset_name: str, split: Optional[str]) -> str:
    cfg = SUPPORTED_DATASETS.get(dataset_name)
    if not cfg:
        supported = ", ".join(SUPPORTED_DATASETS.keys())
        raise SystemExit(f"不支持的数据集: {dataset_name}。可选: {supported}")
    return split or cfg["default_split"]


def normalize_label(value: Any) -> str:
    # 支持数值标签、布尔标签、字符串标签。
    if isinstance(value, bool):
        return "needs_fact_check" if value else "no_fact_check_needed"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) == 0:
            return "no_fact_check_needed"
        if int(value) == 1:
            return "needs_fact_check"

    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    alias = {
        "0": "no_fact_check_needed",
        "1": "needs_fact_check",
        "no fact check needed": "no_fact_check_needed",
        "fact check needed": "needs_fact_check",
        "no_fact_check_needed": "no_fact_check_needed",
        "needs_fact_check": "needs_fact_check",
        "no_fact_check": "no_fact_check_needed",
        "fact_check_needed": "needs_fact_check",
        "no factcheck needed": "no_fact_check_needed",
        "factcheck needed": "needs_fact_check",
    }
    return alias.get(text, text)


def _load_dataset_with_datasets(
    dataset_name: str, split: str, max_samples: int
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=42)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    return [dict(item) for item in ds]


def _load_dataset_with_hf_hub_parquet(
    dataset_name: str, split: str, max_samples: int
) -> List[Dict[str, Any]]:
    from huggingface_hub import snapshot_download
    import pandas as pd

    local_dir = snapshot_download(repo_id=dataset_name, repo_type="dataset")
    parquet_files = sorted(Path(local_dir).rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in dataset snapshot: {dataset_name}")

    split_matches = [p for p in parquet_files if split.lower() in p.name.lower()]
    target_files = split_matches if split_matches else parquet_files

    frames = [pd.read_parquet(p) for p in target_files]
    df = pd.concat(frames, ignore_index=True)
    records = df.to_dict(orient="records")
    random.Random(42).shuffle(records)
    if max_samples > 0:
        records = records[:max_samples]
    return records


def load_dataset_rows(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
    try:
        return _load_dataset_with_datasets(dataset_name, split, max_samples)
    except Exception as exc:
        print(
            "[warn] datasets loader failed, fallback to huggingface_hub+parquet.",
            f"reason: {exc}",
        )
        return _load_dataset_with_hf_hub_parquet(dataset_name, split, max_samples)


def extract_sample_fields(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    cfg = SUPPORTED_DATASETS[dataset_name]
    text = str(sample.get(cfg["text_field"], "") or "").strip()
    expected_raw = sample.get(cfg["label_field"], "")
    expected_norm = normalize_label(expected_raw)
    return {
        "text": text,
        "expected_raw": expected_raw,
        "expected_norm": expected_norm,
        "expected_needs_fact_check": expected_norm == "needs_fact_check",
    }


def print_dataset_stats(dataset_name: str, split: str, samples: List[Dict[str, Any]]) -> None:
    labels: List[str] = []
    text_lengths: List[int] = []

    for sample in samples:
        fields = extract_sample_fields(sample, dataset_name)
        labels.append(fields["expected_norm"])
        text_lengths.append(len(fields["text"]))

    label_counter = Counter(label for label in labels if label)
    avg_text_len = (sum(text_lengths) / len(text_lengths)) if text_lengths else 0.0

    print("[info] selected dataset summary:")
    print(f"  - dataset: {dataset_name}")
    print(f"  - split: {split}")
    print(f"  - sample_count: {len(samples)}")
    print(f"  - unique_labels: {len(label_counter)}")
    print(f"  - avg_text_length: {avg_text_len:.1f}")
    print("  - label_distribution:")
    for label, count in label_counter.most_common():
        print(f"    * {label}: {count}")


def classify_fact_check(router_url: str, text: str, timeout: int) -> Dict[str, Any]:
    url = f"{router_url.rstrip('/')}/api/v1/classify/fact-check"
    response = requests.post(url, json={"text": text}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_response_fields(resp: Dict[str, Any]) -> Dict[str, Any]:
    label = normalize_label(resp.get("label"))
    if not label:
        label = "needs_fact_check" if bool(resp.get("needs_fact_check", False)) else "no_fact_check_needed"
    return {
        "needs_fact_check": bool(resp.get("needs_fact_check", False)),
        "label": label,
        "confidence": resp.get("confidence"),
        "processing_time_ms": resp.get("processing_time_ms"),
    }


def _sanitize_jsonl_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.translate({0x2028: " ", 0x2029: " ", 0x0085: " "})
    if isinstance(value, dict):
        return {key: _sanitize_jsonl_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonl_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_jsonl_value(item) for item in value)
    return value


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(_sanitize_jsonl_value(row), ensure_ascii=False) + "\n")


def compute_percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_one(
    index: int,
    sample: Dict[str, Any],
    dataset_name: str,
    router_url: str,
    timeout: int,
) -> Dict[str, Any]:
    fields = extract_sample_fields(sample, dataset_name)
    text = fields["text"]

    row: Dict[str, Any] = {
        "index": index,
        "dataset": dataset_name,
        "split": sample.get("split"),
        "text": text,
        "expected_raw": fields["expected_raw"],
        "expected_norm": fields["expected_norm"],
        "expected_needs_fact_check": fields["expected_needs_fact_check"],
    }

    try:
        started = time.perf_counter()
        resp = classify_fact_check(router_url, text, timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        parsed = parse_response_fields(resp)

        is_correct = parsed["needs_fact_check"] == fields["expected_needs_fact_check"]

        row.update(
            {
                "status": "ok",
                "predicted_needs_fact_check": parsed["needs_fact_check"],
                "predicted_label": parsed["label"],
                "confidence": parsed["confidence"],
                "processing_time_ms": parsed["processing_time_ms"],
                "latency_ms": latency_ms,
                "is_correct": is_correct,
                "raw_response": resp,
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "predicted_needs_fact_check": None,
                "predicted_label": None,
                "confidence": None,
                "processing_time_ms": None,
                "latency_ms": None,
                "is_correct": None,
                "error": str(exc),
            }
        )

    return row


def main() -> None:
    args = parse_args()
    split = resolve_dataset_split(args.dataset, args.split)

    print("[info] 当前脚本用于 /api/v1/classify/fact-check 的规模化评测。")
    print("[info] 默认数据集为 fact-check-classification-dataset，标签映射为 no_fact_check_needed/needs_fact_check。")

    health_url = f"{args.router_url.rstrip('/')}/health"
    try:
        health_resp = requests.get(health_url, timeout=args.timeout)
        health_resp.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"无法连接 Classification API: {health_url} ({exc})")

    samples = load_dataset_rows(args.dataset, split, args.max_samples)
    total = len(samples)
    print(f"[info] loaded samples: {total}")
    print_dataset_stats(args.dataset, split, samples)

    started = time.time()
    details: List[Dict[str, Any]] = []

    ok_count = 0
    fail_count = 0
    correct = 0
    incorrect = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    expected_positive = 0
    expected_negative = 0
    predicted_positive = 0
    predicted_negative = 0
    api_latencies: List[float] = []
    model_latencies: List[float] = []

    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        for idx, sample in enumerate(samples, start=1):
            row = evaluate_one(idx, sample, args.dataset, args.router_url, args.timeout)
            details.append(row)
            if idx % 100 == 0:
                print(f"[progress] {idx}/{total}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    evaluate_one,
                    idx,
                    sample,
                    args.dataset,
                    args.router_url,
                    args.timeout,
                ): idx
                for idx, sample in enumerate(samples, start=1)
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                details.append(row)
                completed += 1
                if completed % 100 == 0:
                    print(f"[progress] {completed}/{total}")
        details.sort(key=lambda item: item["index"])

    for row in details:
        if row.get("status") != "ok":
            fail_count += 1
            continue

        ok_count += 1
        if isinstance(row.get("latency_ms"), (int, float)):
            api_latencies.append(float(row["latency_ms"]))
        if isinstance(row.get("processing_time_ms"), (int, float)):
            model_latencies.append(float(row["processing_time_ms"]))

        expected = bool(row.get("expected_needs_fact_check"))
        predicted = bool(row.get("predicted_needs_fact_check"))

        if expected:
            expected_positive += 1
        else:
            expected_negative += 1
        if predicted:
            predicted_positive += 1
        else:
            predicted_negative += 1

        if expected == predicted:
            correct += 1
        else:
            incorrect += 1

        if expected and predicted:
            true_positive += 1
        elif not expected and not predicted:
            true_negative += 1
        elif not expected and predicted:
            false_positive += 1
        elif expected and not predicted:
            false_negative += 1

    elapsed = time.time() - started
    evaluated = correct + incorrect
    accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    fact_check_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    fact_check_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )
    avg_api_latency_ms = (
        sum(api_latencies) / len(api_latencies) if api_latencies else None
    )
    avg_processing_time_ms = (
        sum(model_latencies) / len(model_latencies) if model_latencies else None
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    detail_file = out_dir / f"fact_check_eval_detail_{run_id}.jsonl"
    summary_file = out_dir / f"fact_check_eval_summary_{run_id}.json"
    latest_detail = out_dir / "latest_detail.jsonl"
    latest_summary = out_dir / "latest_summary.json"

    write_jsonl(detail_file, details)
    write_jsonl(latest_detail, details)

    summary = {
        "run_id": run_id,
        "router_url": args.router_url,
        "endpoint": "/api/v1/classify/fact-check",
        "dataset": args.dataset,
        "split": split,
        "total_samples": total,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "evaluated": evaluated,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "fact_check_recall": fact_check_recall,
        "fact_check_precision": fact_check_precision,
        "false_positive_rate": false_positive_rate,
        "avg_api_latency_ms": avg_api_latency_ms,
        "avg_processing_time_ms": avg_processing_time_ms,
        "p50_api_latency_ms": compute_percentile(api_latencies, 50),
        "p95_api_latency_ms": compute_percentile(api_latencies, 95),
        "elapsed_seconds": elapsed,
        "detail_file": str(detail_file),
        "note": "Batch performance test for fact-check classification.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    latest_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[info] evaluation completed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()