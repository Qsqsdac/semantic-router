#!/usr/bin/env python3
"""Chunked RouterArena e2e benchmark.

Use this script when upstream API limits prevent running thousands of requests
in one pass. It shuffles samples, slices them into fixed-size chunks, and runs
only one chunk per execution.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import routerarena_e2e_benchmark as base


DEFAULT_SLICE_SIZE = 500
DEFAULT_SLICE_INDEX = 0
DEFAULT_SHUFFLE_SEED = 20260418


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunked RouterArena-backed end-to-end benchmark for /v1/chat/completions"
    )
    parser.add_argument(
        "--router-url",
        default="http://localhost:9099",
        help="OpenAI-compatible router URL",
    )
    parser.add_argument(
        "--endpoint",
        default=base.DEFAULT_ENDPOINT,
        help="Chat completions endpoint path",
    )
    parser.add_argument(
        "--dataset",
        default=base.DEFAULT_DATASET,
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=base.DEFAULT_SPLITS,
        choices=sorted(base.SUPPORTED_SPLITS),
        help="RouterArena splits to evaluate",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=DEFAULT_SLICE_SIZE,
        help="Chunk size after shuffle (default: 500)",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=DEFAULT_SLICE_INDEX,
        help="0-based chunk index to run",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help="Random seed used for deterministic shuffle",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base.DEFAULT_OUTPUT_DIR),
        help="Base directory where detail and summary files are written",
    )
    parser.add_argument(
        "--output-subdir",
        default="chunked",
        help=(
            "Append this leaf subdirectory under output-dir. "
            "Final path = output-dir/output-subdir"
        ),
    )
    parser.add_argument(
        "--auth-token",
        default=base.DEFAULT_AUTH_TOKEN,
        help="Bearer token used in the tutorial example",
    )
    parser.add_argument(
        "--user-id",
        default=base.DEFAULT_USER_ID,
        help="Value for x-authz-user-id",
    )
    parser.add_argument(
        "--user-groups",
        default=base.DEFAULT_USER_GROUPS,
        help="Value for x-authz-user-groups",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help="Optional reasoning_effort value in the request body; empty disables this field.",
    )
    parser.add_argument(
        "--model",
        default=base.DEFAULT_ROUTER_MODEL,
        help="OpenAI-compatible model field used in the tutorial example",
    )
    return parser.parse_args()


def _slice_rows(rows: List[Dict[str, Any]], slice_size: int, slice_index: int) -> Tuple[List[Dict[str, Any]], int]:
    total = len(rows)
    total_slices = max(1, math.ceil(total / slice_size))
    if slice_index < 0 or slice_index >= total_slices:
        raise ValueError(
            f"slice-index {slice_index} out of range [0, {total_slices - 1}] for total={total}, slice-size={slice_size}"
        )

    start = slice_index * slice_size
    end = min(total, start + slice_size)
    return rows[start:end], total_slices


def build_chunked_samples(
    dataset: str,
    splits: List[str],
    slice_size: int,
    slice_index: int,
    shuffle_seed: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]]]:
    # Load all entries first, then apply deterministic shuffle and chunking.
    split_samples = base.load_aligned_splits(dataset, splits, max_samples=0)
    rng = random.Random(shuffle_seed)
    metadata: Dict[str, Dict[str, int]] = {}

    if "full" in split_samples and "robustness" in split_samples:
        full_rows = split_samples["full"]
        robust_rows = split_samples["robustness"]

        full_map = {base._global_index_of(row): row for row in full_rows}
        robust_map = {base._global_index_of(row): row for row in robust_rows}
        common_indices = [
            gid
            for gid in full_map.keys()
            if gid in robust_map and gid
        ]

        rng.shuffle(common_indices)
        selected_indices, total_slices = _slice_rows(common_indices, slice_size, slice_index)

        split_samples["full"] = [full_map[gid] for gid in selected_indices]
        split_samples["robustness"] = [robust_map[gid] for gid in selected_indices]

        metadata["full"] = {
            "total_before_slice": len(full_rows),
            "total_after_slice": len(split_samples["full"]),
            "slice_size": slice_size,
            "slice_index": slice_index,
            "total_slices": total_slices,
            "shuffle_seed": shuffle_seed,
        }
        metadata["robustness"] = {
            "total_before_slice": len(robust_rows),
            "total_after_slice": len(split_samples["robustness"]),
            "slice_size": slice_size,
            "slice_index": slice_index,
            "total_slices": total_slices,
            "shuffle_seed": shuffle_seed,
        }
    else:
        for split in splits:
            rows = split_samples.get(split, [])
            shuffled = list(rows)
            rng.shuffle(shuffled)
            selected_rows, total_slices = _slice_rows(shuffled, slice_size, slice_index)
            split_samples[split] = selected_rows
            metadata[split] = {
                "total_before_slice": len(rows),
                "total_after_slice": len(selected_rows),
                "slice_size": slice_size,
                "slice_index": slice_index,
                "total_slices": total_slices,
                "shuffle_seed": shuffle_seed,
            }

    return split_samples, metadata


def main() -> None:
    args = parse_args()

    if args.slice_size <= 0:
        raise ValueError("slice-size must be > 0")

    output_base = Path(args.output_dir).expanduser()
    if not output_base.is_absolute():
        output_base = (base.REPO_ROOT / output_base).resolve()
    output_dir = output_base / args.output_subdir if args.output_subdir else output_base
    output_dir.mkdir(parents=True, exist_ok=True)

    split_samples, split_slice_meta = build_chunked_samples(
        dataset=args.dataset,
        splits=args.splits,
        slice_size=args.slice_size,
        slice_index=args.slice_index,
        shuffle_seed=args.shuffle_seed,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split in args.splits:
        samples = split_samples.get(split, [])
        meta = split_slice_meta.get(split, {})
        print(
            "[info] split=%s, chunk=%s/%s, samples=%s"
            % (
                split,
                meta.get("slice_index", args.slice_index),
                max(0, meta.get("total_slices", 1) - 1),
                len(samples),
            )
        )

        start = time.time()
        rows: List[Dict[str, Any]] = []
        consecutive_failures = 0

        for index, sample in enumerate(samples, start=1):
            row = base.evaluate_one(
                index=index,
                row=sample,
                router_url=args.router_url,
                endpoint=args.endpoint,
                model=args.model,
                auth_token=args.auth_token,
                user_id=args.user_id,
                user_groups=args.user_groups,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
            rows.append(row)

            if row.get("status") == "ok":
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if index % 100 == 0:
                print(f"[progress] {split}: {index}/{len(samples)}")

            if consecutive_failures >= base.MAX_CONSECUTIVE_FAILURES:
                print(
                    f"[error] split={split}: reached {base.MAX_CONSECUTIVE_FAILURES} consecutive failures, stopping this split early"
                )
                break

        split_rows[split] = rows

        detail_file = output_dir / f"routerarena_e2e_{split}_detail_{run_id}.jsonl"
        latest_detail_file = output_dir / f"latest_{split}_detail.jsonl"
        base.write_jsonl(detail_file, rows)
        base.write_jsonl(latest_detail_file, rows)

        summary = base.summarize_rows(rows)
        summary.update(
            {
                "run_id": run_id,
                "router_url": args.router_url,
                "endpoint": args.endpoint,
                "dataset": args.dataset,
                "split": split,
                "detail_file": str(detail_file),
                "elapsed_seconds": time.time() - start,
                "slice": meta,
            }
        )
        split_summaries[split] = summary

    robustness_summary: Dict[str, Any] = {}
    if "full" in split_rows and "robustness" in split_rows:
        robustness_summary = base.compute_robustness(
            split_rows["full"], split_rows["robustness"]
        )

    combined_summary = {
        "run_id": run_id,
        "router_url": args.router_url,
        "endpoint": args.endpoint,
        "dataset": args.dataset,
        "model": args.model,
        "splits": split_summaries,
        "robustness": robustness_summary,
        "slice": {
            "slice_size": args.slice_size,
            "slice_index": args.slice_index,
            "shuffle_seed": args.shuffle_seed,
        },
        "output_subdir": args.output_subdir,
        "note": (
            "Chunked RouterArena benchmark through /v1/chat/completions. "
            "Samples are deterministically shuffled then sliced."
        ),
    }

    summary_file = output_dir / f"routerarena_e2e_summary_{run_id}.json"
    latest_summary = output_dir / "latest_summary.json"
    summary_file.write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False, default=base.json_default),
        encoding="utf-8",
    )
    latest_summary.write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False, default=base.json_default),
        encoding="utf-8",
    )

    print("[info] benchmark finished")
    print(json.dumps(combined_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
