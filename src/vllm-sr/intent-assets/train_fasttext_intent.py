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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fastText intent model from MMLU-Pro")
    parser.add_argument("--dataset", default=SUPPORTED_DATASET)
    parser.add_argument("--split", default=SUPPORTED_SPLIT)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fasttext-bin", default="./bin/fasttext.real")
    parser.add_argument("--work-dir", default="./.build/fasttext-intent")
    parser.add_argument("--output-model", default="./models/intent_fasttext.bin")

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


def load_with_datasets(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    return [dict(item) for item in ds]


def load_with_hf_hub_parquet(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
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
    if max_samples > 0:
        records = records[:max_samples]
    return records


def load_samples(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
    try:
        return load_with_datasets(dataset_name, split, max_samples)
    except Exception as exc:
        print(f"[warn] datasets loader failed, fallback to huggingface_hub+parquet: {exc}")
        return load_with_hf_hub_parquet(dataset_name, split, max_samples)


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

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_model = Path(args.output_model).resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)

    fasttext_bin = Path(args.fasttext_bin).resolve()
    if not fasttext_bin.exists():
        raise SystemExit(f"fasttext binary not found: {fasttext_bin}")

    samples = load_samples(args.dataset, args.split, args.max_samples)
    lines = [x for x in (to_fasttext_line(s) for s in samples) if x]
    if len(lines) < 100:
        raise SystemExit(f"not enough valid samples: {len(lines)}")

    rnd = random.Random(args.seed)
    rnd.shuffle(lines)

    train_count = int(len(lines) * args.train_ratio)
    train_count = max(1, min(train_count, len(lines) - 1))
    train_lines = lines[:train_count]
    valid_lines = lines[train_count:]

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
