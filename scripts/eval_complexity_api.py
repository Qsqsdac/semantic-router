#!/usr/bin/env python3
"""
面向 Classification API 的 complexity 规模化评测脚本。

说明：
1) 默认使用本地数据集：src/vllm-sr/complexity-assets/.build/complexity/valid.jsonl。
2) 目标接口是 /api/v1/classify/complexity。
3) 保存逐条样本输出和汇总指标，便于回归比较。

示例：
python scripts/eval_complexity_api.py \
  --router-url http://localhost:8080 \
  --workers 4 \
  --max-samples 1000 \
  --output-dir reports/classification-complexity
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_FILE = (
	REPO_ROOT
	/ "src"
	/ "vllm-sr"
	/ "complexity-assets"
	/ ".build"
	/ "complexity"
	/ "valid.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "classification-complexity"
VALID_LABELS = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Dataset-driven benchmark for /api/v1/classify/complexity"
	)
	parser.add_argument(
		"--router-url",
		default="http://localhost:8080",
		help="Classification API base URL (default: http://localhost:8080)",
	)
	parser.add_argument(
		"--dataset-file",
		default=str(DEFAULT_DATASET_FILE),
		help=(
			"Local jsonl dataset file path "
			"(default: src/vllm-sr/complexity-assets/.build/complexity/valid.jsonl)"
		),
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
		"--rule-name",
		default="",
		help=(
			"Rule name to evaluate if response has multiple complexity rules. "
			"Empty means use the first result."
		),
	)
	parser.add_argument(
		"--output-dir",
		default=str(DEFAULT_OUTPUT_DIR),
		help="Directory to save detail/summary files",
	)
	return parser.parse_args()


def normalize_label(value: Any) -> str:
	text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
	alias = {
		"easy": "easy",
		"simple": "easy",
		"low": "easy",
		"medium": "medium",
		"normal": "medium",
		"moderate": "medium",
		"hard": "hard",
		"complex": "hard",
		"difficult": "hard",
	}
	return alias.get(text, text)


def load_local_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
	if not path.exists():
		raise SystemExit(f"数据集文件不存在: {path}")

	rows: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			stripped = line.strip()
			if not stripped:
				continue
			try:
				obj = json.loads(stripped)
			except json.JSONDecodeError as exc:
				raise SystemExit(f"数据集文件 JSON 解析失败: {path}:{line_no} ({exc})")
			if isinstance(obj, dict):
				rows.append(obj)

	if max_samples > 0:
		rows = rows[:max_samples]
	return rows


def extract_sample_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
	text = str(sample.get("request", "") or "").strip()
	expected_raw = sample.get("label", "")
	expected_norm = normalize_label(expected_raw)
	return {
		"text": text,
		"expected_raw": expected_raw,
		"expected_norm": expected_norm,
	}


def print_dataset_stats(dataset_file: Path, samples: List[Dict[str, Any]]) -> None:
	labels: List[str] = []
	text_lengths: List[int] = []

	for sample in samples:
		fields = extract_sample_fields(sample)
		labels.append(fields["expected_norm"])
		text_lengths.append(len(fields["text"]))

	label_counter = Counter(label for label in labels if label)
	avg_text_len = (sum(text_lengths) / len(text_lengths)) if text_lengths else 0.0

	print("[info] selected dataset summary:")
	print(f"  - dataset_file: {dataset_file}")
	print(f"  - sample_count: {len(samples)}")
	print(f"  - unique_labels: {len(label_counter)}")
	print(f"  - avg_text_length: {avg_text_len:.1f}")
	print("  - label_distribution:")
	for label, count in label_counter.most_common():
		print(f"    * {label}: {count}")


def classify_complexity(router_url: str, text: str, timeout: int) -> Dict[str, Any]:
	url = f"{router_url.rstrip('/')}/api/v1/classify/complexity"
	response = requests.post(url, json={"text": text}, timeout=timeout)
	response.raise_for_status()
	return response.json()


def select_target_result(
	results: List[Dict[str, Any]], rule_name: str
) -> Optional[Dict[str, Any]]:
	if not results:
		return None

	target_rule = rule_name.strip()
	if not target_rule:
		return results[0]

	for item in results:
		if str(item.get("rule_name", "")).strip() == target_rule:
			return item
	return None


def parse_response_fields(resp: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
	results = resp.get("results", []) if isinstance(resp, dict) else []
	if not isinstance(results, list):
		results = []

	normalized_results: List[Dict[str, Any]] = []
	for item in results:
		if not isinstance(item, dict):
			continue
		normalized_results.append(
			{
				"rule_name": item.get("rule_name"),
				"classification": normalize_label(item.get("classification")),
				"raw_difference": item.get("raw_difference"),
				"hard_max_similarity": item.get("hard_max_similarity"),
				"easy_max_similarity": item.get("easy_max_similarity"),
				"threshold": item.get("threshold"),
				"signal_source": item.get("signal_source"),
			}
		)

	target = select_target_result(normalized_results, rule_name)
	return {
		"processing_time_ms": resp.get("processing_time_ms") if isinstance(resp, dict) else None,
		"results": normalized_results,
		"selected_rule_name": target.get("rule_name") if target else None,
		"predicted_label": target.get("classification") if target else None,
		"raw_difference": target.get("raw_difference") if target else None,
		"hard_max_similarity": target.get("hard_max_similarity") if target else None,
		"easy_max_similarity": target.get("easy_max_similarity") if target else None,
		"threshold": target.get("threshold") if target else None,
		"signal_source": target.get("signal_source") if target else None,
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
	router_url: str,
	timeout: int,
	rule_name: str,
) -> Dict[str, Any]:
	fields = extract_sample_fields(sample)
	text = fields["text"]

	row: Dict[str, Any] = {
		"index": index,
		"text": text,
		"expected_raw": fields["expected_raw"],
		"expected_norm": fields["expected_norm"],
	}

	try:
		started = time.perf_counter()
		resp = classify_complexity(router_url, text, timeout)
		latency_ms = (time.perf_counter() - started) * 1000.0
		parsed = parse_response_fields(resp, rule_name)

		predicted = normalize_label(parsed["predicted_label"])
		is_correct = predicted == fields["expected_norm"]

		row.update(
			{
				"status": "ok",
				"predicted_label": predicted,
				"selected_rule_name": parsed["selected_rule_name"],
				"raw_difference": parsed["raw_difference"],
				"hard_max_similarity": parsed["hard_max_similarity"],
				"easy_max_similarity": parsed["easy_max_similarity"],
				"threshold": parsed["threshold"],
				"signal_source": parsed["signal_source"],
				"processing_time_ms": parsed["processing_time_ms"],
				"latency_ms": latency_ms,
				"is_correct": is_correct,
				"result_count": len(parsed["results"]),
				"all_results": parsed["results"],
				"raw_response": resp,
			}
		)
	except Exception as exc:
		row.update(
			{
				"status": "error",
				"predicted_label": None,
				"selected_rule_name": None,
				"raw_difference": None,
				"hard_max_similarity": None,
				"easy_max_similarity": None,
				"threshold": None,
				"signal_source": None,
				"processing_time_ms": None,
				"latency_ms": None,
				"is_correct": None,
				"result_count": 0,
				"all_results": [],
				"error": str(exc),
			}
		)

	return row


def main() -> None:
	args = parse_args()

	dataset_file = Path(args.dataset_file).expanduser()
	if not dataset_file.is_absolute():
		dataset_file = (REPO_ROOT / dataset_file).resolve()

	print("[info] 当前脚本用于 /api/v1/classify/complexity 的规模化评测。")
	print(f"[info] 默认测试集: {DEFAULT_DATASET_FILE}")

	health_url = f"{args.router_url.rstrip('/')}/health"
	try:
		health_resp = requests.get(health_url, timeout=args.timeout)
		health_resp.raise_for_status()
	except Exception as exc:
		raise SystemExit(f"无法连接 Classification API: {health_url} ({exc})")

	samples = load_local_jsonl(dataset_file, args.max_samples)
	total = len(samples)
	print(f"[info] loaded samples: {total}")
	print_dataset_stats(dataset_file, samples)

	started = time.time()
	details: List[Dict[str, Any]] = []

	worker_count = max(1, int(args.workers))
	if worker_count == 1:
		for idx, sample in enumerate(samples, start=1):
			row = evaluate_one(idx, sample, args.router_url, args.timeout, args.rule_name)
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
					args.router_url,
					args.timeout,
					args.rule_name,
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

	ok_count = 0
	fail_count = 0
	correct = 0
	incorrect = 0
	expected_counter: Counter[str] = Counter()
	predicted_counter: Counter[str] = Counter()
	confusion: Dict[str, Dict[str, int]] = {
		label: {inner: 0 for inner in VALID_LABELS}
		for label in VALID_LABELS
	}
	api_latencies: List[float] = []
	model_latencies: List[float] = []

	for row in details:
		if row.get("status") != "ok":
			fail_count += 1
			continue

		ok_count += 1
		if isinstance(row.get("latency_ms"), (int, float)):
			api_latencies.append(float(row["latency_ms"]))
		if isinstance(row.get("processing_time_ms"), (int, float)):
			model_latencies.append(float(row["processing_time_ms"]))

		expected = normalize_label(row.get("expected_norm"))
		predicted = normalize_label(row.get("predicted_label"))

		if expected in VALID_LABELS:
			expected_counter[expected] += 1
		if predicted in VALID_LABELS:
			predicted_counter[predicted] += 1
		if expected in VALID_LABELS and predicted in VALID_LABELS:
			confusion[expected][predicted] += 1

		if expected == predicted:
			correct += 1
		else:
			incorrect += 1

	elapsed = time.time() - started
	evaluated = correct + incorrect
	accuracy = (correct / evaluated) if evaluated > 0 else 0.0
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
	detail_file = out_dir / f"complexity_eval_detail_{run_id}.jsonl"
	summary_file = out_dir / f"complexity_eval_summary_{run_id}.json"
	latest_detail = out_dir / "latest_detail.jsonl"
	latest_summary = out_dir / "latest_summary.json"

	write_jsonl(detail_file, details)
	write_jsonl(latest_detail, details)

	summary = {
		"run_id": run_id,
		"router_url": args.router_url,
		"endpoint": "/api/v1/classify/complexity",
		"dataset_file": str(dataset_file),
		"rule_name": args.rule_name or None,
		"total_samples": total,
		"ok_count": ok_count,
		"fail_count": fail_count,
		"evaluated": evaluated,
		"correct": correct,
		"incorrect": incorrect,
		"accuracy": accuracy,
		"expected_distribution": dict(expected_counter),
		"predicted_distribution": dict(predicted_counter),
		"confusion_matrix": confusion,
		"avg_api_latency_ms": avg_api_latency_ms,
		"avg_processing_time_ms": avg_processing_time_ms,
		"p50_api_latency_ms": compute_percentile(api_latencies, 50),
		"p95_api_latency_ms": compute_percentile(api_latencies, 95),
		"elapsed_seconds": elapsed,
		"detail_file": str(detail_file),
		"note": "Batch performance test for complexity classification.",
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
