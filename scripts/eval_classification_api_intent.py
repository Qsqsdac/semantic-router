#!/usr/bin/env python3
"""
面向 Classification API(8080) 的规模化评测脚本（Intent/Domain）。

说明：
1) 本脚本默认使用 TIGER-Lab/MMLU-Pro 数据集做评测。
2) 当前评测目标是 /api/v1/classify/intent（领域/意图分类）。
3) 会保存逐条样本输出（含序号、预测结果、原始响应）和汇总指标，方便后续模型版本对比。

示例：
python scripts/eval_classification_api_intent.py \
  --router-url http://localhost:8080 \
  --max-samples 2000 \
  --output-dir reports/classification-intent
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_DATASET = "TIGER-Lab/MMLU-Pro"
SUPPORTED_DATASETS: Dict[str, Dict[str, str]] = {
    "TIGER-Lab/MMLU-Pro": {
        "default_split": "test",
        "text_field": "question",
        "label_field": "category",
    },
    "LLM-Semantic-Router/category-classifier-supplement": {
        "default_split": "train",
        "text_field": "text",
        "label_field": "label",
    },
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "classification-intent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-driven benchmark for /api/v1/classify/intent"
    )
    parser.add_argument(
        "--router-url",
        default="http://localhost:8080",
        help="Classification API base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "HF dataset for intent benchmark "
            f"(default: {DEFAULT_DATASET}, supported: {', '.join(SUPPORTED_DATASETS)})"
        ),
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
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save detail/summary files",
    )
    return parser.parse_args()


def normalize_category(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    alias = {
        "computer science": "computer science",
        "math": "math",
        "mathematics": "math",
        "economics": "economics",
        "biology": "biology",
        "chemistry": "chemistry",
        "physics": "physics",
        "history": "history",
        "law": "law",
        "health": "health",
        "engineering": "engineering",
        "philosophy": "philosophy",
        "psychology": "psychology",
        "business": "business",
        "other": "other",
    }
    return alias.get(text, text)


def resolve_dataset_split(dataset_name: str, split: Optional[str]) -> str:
    cfg = SUPPORTED_DATASETS.get(dataset_name)
    if not cfg:
        supported = ", ".join(SUPPORTED_DATASETS.keys())
        raise SystemExit(f"不支持的数据集: {dataset_name}。可选: {supported}")
    return split or cfg["default_split"]


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
    """Fallback loader when `datasets` import fails due to version mismatch.

    Strategy:
    1) Download dataset snapshot from HF Hub.
    2) Read parquet files with pandas.
    3) Select files that match split name when possible.
    """
    from huggingface_hub import snapshot_download
    import pandas as pd

    local_dir = snapshot_download(repo_id=dataset_name, repo_type="dataset")
    parquet_files = sorted(Path(local_dir).rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in dataset snapshot: {dataset_name}")

    # Prefer split-matching files (e.g., test-*.parquet)
    split_matches = [p for p in parquet_files if split.lower() in p.name.lower()]
    target_files = split_matches if split_matches else parquet_files

    frames = [pd.read_parquet(p) for p in target_files]
    df = pd.concat(frames, ignore_index=True)

    records = df.to_dict(orient="records")
    random.Random(42).shuffle(records)
    if max_samples > 0:
        records = records[:max_samples]
    return records


def load_hf_dataset(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
    try:
        return _load_dataset_with_datasets(dataset_name, split, max_samples)
    except Exception as exc:
        print(
            "[warn] datasets loader failed, fallback to huggingface_hub+parquet.",
            f"reason: {exc}",
        )
        return _load_dataset_with_hf_hub_parquet(dataset_name, split, max_samples)


def extract_eval_fields(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    cfg = SUPPORTED_DATASETS[dataset_name]
    text = str(sample.get(cfg["text_field"], "")).strip()
    expected_raw = sample.get(cfg["label_field"], "")
    return {
        "text": text,
        "expected_raw": expected_raw,
        "expected_norm": normalize_category(expected_raw),
    }


def print_dataset_stats(dataset_name: str, split: str, samples: List[Dict[str, Any]]) -> None:
    labels: List[str] = []
    text_lengths: List[int] = []

    for sample in samples:
        fields = extract_eval_fields(sample, dataset_name)
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


def classify_intent(router_url: str, text: str, timeout: int) -> Dict[str, Any]:
    url = f"{router_url.rstrip('/')}/api/v1/classify/intent"
    response = requests.post(url, json={"text": text}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_response_fields(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stable fields from classify/intent response."""
    cls = resp.get("classification", {}) if isinstance(resp, dict) else {}
    matched = (
        resp.get("matched_signals", {}).get("domains", [])
        if isinstance(resp, dict)
        else []
    )

    predicted_raw = cls.get("category")
    if not predicted_raw and isinstance(matched, list) and matched:
        predicted_raw = matched[0]

    return {
        "predicted_raw": predicted_raw,
        "predicted_norm": normalize_category(predicted_raw),
        "confidence": cls.get("confidence"),
        "processing_time_ms": cls.get("processing_time_ms"),
        "recommended_model": resp.get("recommended_model"),
        "routing_decision": resp.get("routing_decision"),
        "matched_domains": matched,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    split = resolve_dataset_split(args.dataset, args.split)

    print("[info] 当前脚本用于 /api/v1/classify/intent 的规模化评测。")
    print("[info] 非 intent 任务（如 pii/security/fact-check）后续可独立扩展。")

    # health check
    health_url = f"{args.router_url.rstrip('/')}/health"
    try:
        health_resp = requests.get(health_url, timeout=args.timeout)
        health_resp.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"无法连接 Classification API: {health_url} ({exc})")

    samples = load_hf_dataset(args.dataset, split, args.max_samples)
    total = len(samples)
    print(f"[info] loaded samples: {total}")
    print_dataset_stats(args.dataset, split, samples)

    started = time.time()
    details: List[Dict[str, Any]] = []

    ok_count = 0
    fail_count = 0
    correct = 0
    incorrect = 0
    processing_times: List[float] = []

    for idx, sample in enumerate(samples, start=1):
        fields = extract_eval_fields(sample, args.dataset)
        question = fields["text"]
        expected_raw = fields["expected_raw"]
        expected_norm = fields["expected_norm"]

        row: Dict[str, Any] = {
            "index": idx,
            "dataset": args.dataset,
            "split": split,
            "question": question,
            "expected_raw": expected_raw,
            "expected_norm": expected_norm,
        }

        try:
            resp = classify_intent(args.router_url, question, args.timeout)
            parsed = parse_response_fields(resp)

            is_correct = expected_norm in parsed["matched_domains"]
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            ok_count += 1

            pt = parsed.get("processing_time_ms")
            if isinstance(pt, (int, float)):
                processing_times.append(float(pt))

            row.update(
                {
                    "status": "ok",
                    "predicted_raw": parsed["predicted_raw"],
                    "predicted_norm": parsed["predicted_norm"],
                    "confidence": parsed["confidence"],
                    "processing_time_ms": parsed["processing_time_ms"],
                    "recommended_model": parsed["recommended_model"],
                    "routing_decision": parsed["routing_decision"],
                    "matched_domains": parsed["matched_domains"],
                    "is_correct": is_correct,
                    "raw_response": resp,
                }
            )
        except Exception as exc:
            fail_count += 1
            row.update(
                {
                    "status": "error",
                    "predicted_raw": None,
                    "predicted_norm": "",
                    "confidence": None,
                    "processing_time_ms": None,
                    "recommended_model": None,
                    "routing_decision": None,
                    "matched_domains": [],
                    "is_correct": None,
                    "error": str(exc),
                }
            )

        details.append(row)

        if idx % 100 == 0:
            print(f"[progress] {idx}/{total}")

    elapsed = time.time() - started
    evaluated = correct + incorrect
    accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    avg_processing_time_ms = (
        sum(processing_times) / len(processing_times) if processing_times else None
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    detail_file = out_dir / f"intent_eval_detail_{run_id}.jsonl"
    summary_file = out_dir / f"intent_eval_summary_{run_id}.json"
    latest_detail = out_dir / "latest_detail.jsonl"
    latest_summary = out_dir / "latest_summary.json"

    write_jsonl(detail_file, details)
    write_jsonl(latest_detail, details)

    summary = {
        "run_id": run_id,
        "router_url": args.router_url,
        "endpoint": "/api/v1/classify/intent",
        "dataset": args.dataset,
        "split": split,
        "total_samples": total,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "evaluated": evaluated,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "avg_processing_time_ms": avg_processing_time_ms,
        "elapsed_seconds": elapsed,
        "note": "当前脚本仅用于 intent/domain 分类评测。",
        "detail_file": str(detail_file),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Benchmark Done ===")
    print(f"run_id: {run_id}")
    print(f"dataset: {args.dataset} ({split})")
    print(f"accuracy: {accuracy:.4f} ({correct}/{evaluated})")
    if avg_processing_time_ms is not None:
        print(f"avg_processing_time_ms: {avg_processing_time_ms:.2f}")
    print(f"api errors: {fail_count}")
    print(f"detail: {detail_file}")
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
