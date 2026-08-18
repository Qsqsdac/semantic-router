#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List


SUPPORTED_DATASET = "TIGER-Lab/MMLU-Pro"
SUPPORTED_SPLIT = "test"
SUPPORTED_VALID_SPLIT = "validation"
HOLDOUT_SIZE = 1000
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fastText intent model from MMLU-Pro")
    parser.add_argument("--dataset", default=SUPPORTED_DATASET)
    parser.add_argument("--split", default=SUPPORTED_SPLIT)
    parser.add_argument("--max-samples", type=int, default=0, help="train sample limit after holdout; 0 means all remaining")
    parser.add_argument("--holdout", type=int, default=HOLDOUT_SIZE, help="first N shuffled records reserved for evaluation")
    parser.add_argument("--valid-split", default=SUPPORTED_VALID_SPLIT, help="split used as fastText validation set")
    parser.add_argument("--fasttext-bin", default="bin/fasttext.real")
    parser.add_argument("--work-dir", default=".build/fasttext-intent")
    parser.add_argument("--output-model", default="models/intent_fasttext.bin")

    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--minn", type=int, default=2)
    parser.add_argument("--maxn", type=int, default=5)
    parser.add_argument("--loss", default="hs", choices=["softmax", "hs", "ova"])

    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    alias = {
        "computer science": "computer_science",
        "mathematics": "math",
        "math": "math",
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
    return alias.get(text, text.replace(" ", "_"))


def _locate_local_hf_hub_parquet(dataset_name: str) -> List[Path]:
    """Locate dataset parquet files already cached in the local HF Hub cache.

    Offline-safe and deterministic: the local cache is the canonical source used
    by both this script and scripts/eval_intent_api.py, so both share the exact
    same shuffled record order (fixed seed below).
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return []
    repo_cache_dir = Path(HF_HUB_CACHE) / ("datasets--" + dataset_name.replace("/", "--"))
    if not repo_cache_dir.exists():
        return []
    for snapshot in sorted(repo_cache_dir.glob("snapshots/*")):
        parquet_files = sorted(snapshot.rglob("*.parquet"))
        if parquet_files:
            return parquet_files
    return []


def _read_split_parquet(parquet_files: List[Path], split: str) -> List[Dict[str, Any]]:
    import pandas as pd

    split_matches = [p for p in parquet_files if split.lower() in p.name.lower()]
    target_files = split_matches if split_matches else parquet_files
    frames = [pd.read_parquet(p) for p in target_files]
    df = pd.concat(frames, ignore_index=True)
    return df.to_dict(orient="records")


def load_samples(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    """Load a full split, shuffled with a fixed seed.

    The fixed shuffle (random.Random(42)) MUST match scripts/eval_intent_api.py
    so that the first `holdout` records are identical in both scripts: training
    uses the tail, evaluation uses the head.
    """
    local_parquet_files = _locate_local_hf_hub_parquet(dataset_name)
    if local_parquet_files:
        records = _read_split_parquet(local_parquet_files, split)
        random.Random(42).shuffle(records)
        return records

    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        records = [dict(item) for item in ds]
        random.Random(42).shuffle(records)
        return records
    except Exception as exc:
        print(f"[warn] datasets loader failed, fallback to huggingface_hub+parquet: {exc}")
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=dataset_name, repo_type="dataset")
        parquet_files = sorted(Path(local_dir).rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in dataset snapshot: {dataset_name}")
        records = _read_split_parquet(parquet_files, split)
        random.Random(42).shuffle(records)
        return records


def to_fasttext_line(sample: Dict[str, Any]) -> str | None:
    label = normalize_label(sample.get("category", ""))
    question = str(sample.get("question", "")).strip().replace("\n", " ")
    if not label or not question:
        return None
    return f"__label__{label} {question}"


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    work_dir = resolve_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_model = resolve_path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    fasttext_bin = resolve_path(args.fasttext_bin)
    if not fasttext_bin.exists():
        raise SystemExit(f"fasttext binary not found: {fasttext_bin}")

    test_records = load_samples(args.dataset, args.split)
    if args.holdout >= len(test_records):
        raise SystemExit(
            f"holdout ({args.holdout}) must be smaller than dataset size ({len(test_records)})"
        )
    train_records = test_records[args.holdout:]
    if args.max_samples > 0:
        train_records = train_records[: args.max_samples]

    valid_records = load_samples(args.dataset, args.valid_split)

    train_lines = [x for x in (to_fasttext_line(s) for s in train_records) if x]
    valid_lines = [x for x in (to_fasttext_line(s) for s in valid_records) if x]
    if len(train_lines) < 100:
        raise SystemExit(f"not enough valid train samples: {len(train_lines)}")
    if not valid_lines:
        raise SystemExit(
            f"no valid samples in split {args.valid_split!r} (got {len(valid_records)} records)"
        )

    train_path = work_dir / "train.txt"
    valid_path = work_dir / "valid.txt"
    write_lines(train_path, train_lines)
    write_lines(valid_path, valid_lines)

    model_prefix = work_dir / "intent_fasttext"

    run([
        str(fasttext_bin),
        "supervised",
        "-input", str(train_path),
        "-output", str(model_prefix),
        "-lr", str(args.lr),
        "-epoch", str(args.epoch),
        "-wordNgrams", str(args.word_ngrams),
        "-dim", str(args.dim),
        "-minn", str(args.minn),
        "-maxn", str(args.maxn),
        "-loss", str(args.loss),
        "-thread", str(os.cpu_count() or 2),
    ])

    run([
        str(fasttext_bin),
        "test",
        str(model_prefix) + ".bin",
        str(valid_path),
    ])

    bin_model = Path(str(model_prefix) + ".bin")
    if not bin_model.exists():
        raise SystemExit(f"expected model not found: {bin_model}")

    output_model.write_bytes(bin_model.read_bytes())
    print(f"[ok] model saved: {output_model}")
    print("[next] keep config path at /app/intent-assets/models/intent_fasttext.bin and restart stack")


if __name__ == "__main__":
    main()
