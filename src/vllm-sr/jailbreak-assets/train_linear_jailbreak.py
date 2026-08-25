#!/usr/bin/env python3
"""Train and export the portable linear token model used by jailbreak L1."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DATASET = "llm-semantic-router/jailbreak-detection-dataset"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a CountVectorizer + LogisticRegression jailbreak L1")
	parser.add_argument("--dataset", default=DATASET)
	parser.add_argument("--train-split", default="train")
	parser.add_argument("--validation-split", default="validation")
	parser.add_argument("--max-features", type=int, default=12000)
	parser.add_argument("--min-df", type=int, default=2)
	parser.add_argument("--c", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-model", default="../models/jailbreak_linear_token.json")
	parser.add_argument("--unsafe-threshold", type=float, default=0.90)
	parser.add_argument("--benign-threshold", type=float, default=0.05)
	parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
	return parser.parse_args()


def normalize_label(value: Any) -> str:
	if isinstance(value, bool):
		return "unsafe" if value else "safe"
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		return "unsafe" if int(value) == 1 else "safe"
	text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
	if text in {"1", "unsafe", "jailbreak", "malicious"}:
		return "unsafe"
	return "safe"


def load_rows(dataset: str, split: str, seed: int) -> list[tuple[str, str]]:
	from datasets import load_dataset

	records = load_dataset(dataset, split=split).shuffle(seed=seed)
	rows = []
	for record in records:
		text = str(record.get("text", record.get("prompt", "")) or "").strip().replace("\n", " ")
		label = normalize_label(record.get("label", record.get("label_id", "")))
		if text:
			rows.append((text, label))
	return rows


def export_model(vectorizer: CountVectorizer, classifier: LogisticRegression, path: Path) -> None:
	classes = list(classifier.classes_)
	unsafe_index = classes.index("unsafe")
	if unsafe_index != 1:
		raise RuntimeError(f"expected unsafe to be positive class, got classes={classes}")
	vocabulary = vectorizer.vocabulary_
	tokens = [""] * len(vocabulary)
	for token, index in vocabulary.items():
		tokens[index] = token
	weights = [float(value) for value in classifier.coef_[0].tolist()]
	payload = {
		"model_type": "countvectorizer_binary_unigram_logistic_regression",
		"tokens": tokens,
		"weights": weights,
		"intercept": float(classifier.intercept_[0]),
		"negative_label": "benign",
		"positive_label": "jailbreak",
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def report_gate_metrics(probabilities: list[float], labels: list[str], unsafe_threshold: float, benign_threshold: float) -> None:
	unsafe = [label == "unsafe" for label in labels]
	unsafe_exits = [score >= unsafe_threshold for score in probabilities]
	benign_exits = [score <= benign_threshold for score in probabilities]
	print(
		"[gate] "
		f"unsafe_exit={sum(unsafe_exits)} "
		f"unsafe_precision={sum(exit_ and target for exit_, target in zip(unsafe_exits, unsafe)) / max(1, sum(unsafe_exits)):.4f} "
		f"benign_exit={sum(benign_exits)} "
		f"unsafe_leak={sum(exit_ and target for exit_, target in zip(benign_exits, unsafe)) / max(1, sum(benign_exits)):.4f}"
	)


def main() -> None:
	args = parse_args()
	if not 0 < args.benign_threshold < args.unsafe_threshold < 1:
		raise SystemExit("thresholds must satisfy 0 < benign < unsafe < 1")
	os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
	train_rows = load_rows(args.dataset, args.train_split, args.seed)
	valid_rows = load_rows(args.dataset, args.validation_split, args.seed)
	if len(train_rows) < 100 or not valid_rows:
		raise SystemExit(f"insufficient rows: train={len(train_rows)} validation={len(valid_rows)}")
	random.Random(args.seed).shuffle(train_rows)
	x_train, y_train = zip(*train_rows)
	x_valid, y_valid = zip(*valid_rows)
	vectorizer = CountVectorizer(binary=True, lowercase=True, ngram_range=(1, 1), min_df=args.min_df, max_features=args.max_features)
	x_train_features = vectorizer.fit_transform(x_train)
	classifier = LogisticRegression(C=args.c, max_iter=2000, class_weight="balanced", random_state=args.seed)
	classifier.fit(x_train_features, y_train)
	probabilities = classifier.predict_proba(vectorizer.transform(x_valid))[:, list(classifier.classes_).index("unsafe")].tolist()
	predictions = ["unsafe" if score >= 0.5 else "safe" for score in probabilities]
	precision, recall, f1, _ = precision_recall_fscore_support(y_valid, predictions, pos_label="unsafe", average="binary", zero_division=0)
	print(f"[eval] train={len(train_rows)} validation={len(valid_rows)} accuracy={accuracy_score(y_valid, predictions):.4f} unsafe_precision={precision:.4f} unsafe_recall={recall:.4f} unsafe_f1={f1:.4f}")
	report_gate_metrics(probabilities, list(y_valid), args.unsafe_threshold, args.benign_threshold)
	output = (SCRIPT_DIR / args.output_model).resolve()
	export_model(vectorizer, classifier, output)
	print(f"[ok] model saved: {output}")


if __name__ == "__main__":
	main()