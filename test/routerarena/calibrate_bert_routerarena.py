#!/usr/bin/env python3
"""Calibrate RouteLLM BERT scores on a stratified RouterArena sample."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import routerarena_e2e_benchmark as base

DEFAULT_CHECKPOINT = "routellm/bert_gpt4_augmented"
DEFAULT_OUTPUT = base.REPO_ROOT / "reports" / "routerarena-e2e" / "routellm" / "bert-calibration"
DEFAULT_SEED = 20260915
DEFAULT_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=base.DEFAULT_DATASET)
    parser.add_argument("--split", default="full", choices=sorted(base.SUPPORTED_SPLITS))
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def proportional_counts(groups: Dict[str, List[Dict[str, Any]]], sample_size: int) -> Dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    if sample_size < 1 or sample_size > total:
        raise ValueError(f"sample-size must be between 1 and {total}")

    raw = {name: len(rows) * sample_size / total for name, rows in groups.items()}
    counts = {name: min(len(groups[name]), math.floor(value)) for name, value in raw.items()}
    remaining = sample_size - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def stratified_sample(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[base.resolve_dataset_name(row)].append(row)

    rng = random.Random(seed)
    counts = proportional_counts(groups, sample_size)
    selected: List[Dict[str, Any]] = []
    for name in sorted(groups):
        candidates = list(groups[name])
        rng.shuffle(candidates)
        selected.extend(candidates[:counts[name]])
    rng.shuffle(selected)
    return selected


def bert_scores(
    prompts: List[str], checkpoint: str, batch_size: int
) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    model.eval()
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            inputs = tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            logits = model(**inputs).logits.detach().cpu().numpy()
            logits -= np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            scores.extend((1.0 - np.sum(probabilities[:, -2:], axis=1)).tolist())
    return scores


def quantile(values: List[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    rows = base.load_routerarena_split(args.dataset, args.split, max_samples=0)
    selected = stratified_sample(rows, args.sample_size, args.seed)
    prompts = [base.build_sample_prompt(row)[2] for row in selected]
    scores = bert_scores(prompts, args.checkpoint, args.batch_size)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (base.REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / f"bert_{args.split}_{args.sample_size}_seed{args.seed}_scores.jsonl"
    with detail_path.open("w", encoding="utf-8") as handle:
        for index, (row, score) in enumerate(zip(selected, scores), start=1):
            dataset_name, metric_name, prompt = base.build_sample_prompt(row)
            handle.write(json.dumps({
                "index": index,
                "global_index": base._global_index_of(row),
                "dataset_name": dataset_name,
                "metric_name": metric_name,
                "bert_score": score,
                "prompt": prompt,
            }, ensure_ascii=False) + "\n")

    source_counts = Counter(base.resolve_dataset_name(row) for row in selected)
    values = [float(score) for score in scores]
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "population_size": len(rows),
        "sample_size": len(selected),
        "source_count": len(source_counts),
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "source_counts": dict(sorted(source_counts.items())),
        "score_range": {"min": min(values), "max": max(values)},
        "thresholds_by_strong_ratio": {
            str(ratio): quantile(values, 1.0 - ratio)
            for ratio in (0.10, 0.15, 0.1681, 0.20, 0.30, 0.3524, 0.35, 0.50)
        },
        "detail_file": str(detail_path),
    }
    summary_path = output_dir / f"bert_{args.split}_{args.sample_size}_seed{args.seed}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
