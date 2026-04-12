#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


SUPPORTED_DATASET = "llm-semantic-router/fact-check-classification-dataset"
SUPPORTED_SPLIT = "train"
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
	path = Path(value)
	if path.is_absolute():
		return path
	return (SCRIPT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train fact-check classifier from HF dataset"
	)
	parser.add_argument("--dataset", default=SUPPORTED_DATASET)
	parser.add_argument("--split", default=SUPPORTED_SPLIT)
	parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
	parser.add_argument("--train-ratio", type=float, default=0.9)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--work-dir", default=".build/fact-check")
	parser.add_argument("--output-model", default="models/fact_check_svc.joblib")

	parser.add_argument("--max-features", type=int, default=5000)
	parser.add_argument("--min-df", type=int, default=3)
	parser.add_argument("--max-df", type=float, default=0.95)
	parser.add_argument("--c", type=float, default=1.0)
	parser.add_argument("--kernel", default="linear", choices=["linear", "rbf"])

	parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
	return parser.parse_args()


def normalize_label(value: Any) -> str:
	if isinstance(value, bool):
		return "needs_fact_check" if value else "no_fact_check_needed"
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		if int(value) == 1:
			return "needs_fact_check"
		if int(value) == 0:
			return "no_fact_check_needed"

	text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
	alias = {
		"0": "no_fact_check_needed",
		"1": "needs_fact_check",
		"no fact check needed": "no_fact_check_needed",
		"fact check needed": "needs_fact_check",
		"no factcheck needed": "no_fact_check_needed",
		"factcheck needed": "needs_fact_check",
		"no_fact_check_needed": "no_fact_check_needed",
		"needs_fact_check": "needs_fact_check",
		"no_fact_check": "no_fact_check_needed",
		"fact_check_needed": "needs_fact_check",
	}
	return alias.get(text, text.replace(" ", "_"))


def load_with_datasets(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
	from datasets import load_dataset

	ds = load_dataset(dataset_name, split=split)
	ds = ds.shuffle(seed=42)
	if max_samples > 0:
		max_samples = min(max_samples, len(ds))
		ds = ds.select(range(len(ds) - max_samples, len(ds)))
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
	random.Random(42).shuffle(records)
	if max_samples > 0:
		max_samples = min(max_samples, len(records))
		records = records[len(records) - max_samples :]
	return records


def load_samples(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
	try:
		return load_with_datasets(dataset_name, split, max_samples)
	except Exception as exc:
		print(f"[warn] datasets loader failed, fallback to huggingface_hub+parquet: {exc}")
		return load_with_hf_hub_parquet(dataset_name, split, max_samples)


def extract_text_and_label(sample: Dict[str, Any]) -> Tuple[str, str] | None:
	text = str(sample.get("text", "") or "").strip().replace("\n", " ")
	label = normalize_label(sample.get("label_id", sample.get("label", "")))
	if not text or not label:
		return None
	return text, label


def build_pipeline(args: argparse.Namespace) -> Pipeline:
	return Pipeline(
		[
			(
				"ngrams",
				CountVectorizer(
					ngram_range=(1, 1),
					lowercase=True,
					max_df=args.max_df,
					min_df=args.min_df,
					max_features=args.max_features,
					binary=True,
				),
			),
			(
				"clf",
				SVC(
					C=args.c,
					gamma="scale",
					kernel=args.kernel,
					random_state=args.seed,
				),
			),
		]
	)


def main() -> None:
	args = parse_args()

	if args.dataset != SUPPORTED_DATASET:
		raise SystemExit(f"Only supported dataset is: {SUPPORTED_DATASET}")

	os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

	work_dir = resolve_path(args.work_dir)
	work_dir.mkdir(parents=True, exist_ok=True)
	output_model = resolve_path(args.output_model)
	output_model.parent.mkdir(parents=True, exist_ok=True)

	samples = load_samples(args.dataset, args.split, args.max_samples)
	rows = [x for x in (extract_text_and_label(s) for s in samples) if x]
	if len(rows) < 100:
		raise SystemExit(f"not enough valid samples: {len(rows)}")

	rnd = random.Random(args.seed)
	rnd.shuffle(rows)

	train_count = int(len(rows) * args.train_ratio)
	train_count = max(1, min(train_count, len(rows) - 1))

	train_rows = rows[:train_count]
	valid_rows = rows[train_count:]

	x_train = [text for text, _ in train_rows]
	y_train = [label for _, label in train_rows]
	x_valid = [text for text, _ in valid_rows]
	y_valid = [label for _, label in valid_rows]

	pipeline = build_pipeline(args)
	pipeline.fit(x_train, y_train)

	y_pred = pipeline.predict(x_valid)
	acc = accuracy_score(y_valid, y_pred)
	f1 = f1_score(y_valid, y_pred, pos_label="needs_fact_check")
	print(f"[eval] samples={len(rows)} train={len(train_rows)} valid={len(valid_rows)}")
	print(f"[eval] accuracy={acc:.4f} f1(needs_fact_check)={f1:.4f}")

	import joblib

	joblib.dump(pipeline, output_model)
	print(f"[ok] model saved: {output_model}")


if __name__ == "__main__":
	main()
