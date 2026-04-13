#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import importlib
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple


SUPPORTED_DATASET = "lmsys/chatbot_arena_conversations"
DEFAULT_SPLIT = "train"
DEFAULT_SAMPLE_SIZE = 11000
DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_FALLBACK_DATASETS = "HuggingFaceH4/ultrachat_200k"
DEFAULT_EASY_MAX = 127
DEFAULT_HARD_MIN = 513
SCRIPT_DIR = Path(__file__).resolve().parent

REQUEST_KEYS = (
	"request",
	"prompt",
	"question",
	"instruction",
	"query",
	"input",
	"text",
)

RESPONSE_KEYS = (
	"response",
	"answer",
	"output",
	"completion",
	"assistant",
	"chosen_response",
	"chosen",
)

CONVERSATION_KEYS = (
	"conversation",
	"messages",
	"dialogue",
	"turns",
	"conversation_a",
	"conversation_b",
)

ROLE_USER = {"user", "human", "prompt", "customer", "client"}
ROLE_ASSISTANT = {"assistant", "model", "gpt", "bot", "agent"}

TOKEN_SPLIT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def resolve_path(value: str) -> Path:
	path = Path(value)
	if path.is_absolute():
		return path
	return (SCRIPT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build easy/medium/hard complexity dataset from Chatbot Arena"
	)
	parser.add_argument("--dataset", default=SUPPORTED_DATASET)
	parser.add_argument("--split", default=DEFAULT_SPLIT)
	parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
	parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-dir", default=".build/complexity")
	parser.add_argument("--train-file", default="train.jsonl")
	parser.add_argument("--valid-file", default="valid.jsonl")
	parser.add_argument("--tokenizer", default="auto")
	parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
	parser.add_argument(
		"--easy-max",
		type=int,
		default=DEFAULT_EASY_MAX,
		help="easy when output token count <= this value",
	)
	parser.add_argument(
		"--hard-min",
		type=int,
		default=DEFAULT_HARD_MIN,
		help="hard when output token count >= this value",
	)
	parser.add_argument(
		"--fallback-datasets",
		default=DEFAULT_FALLBACK_DATASETS,
		help="Comma-separated fallback datasets when primary dataset is unavailable",
	)
	return parser.parse_args()


def find_local_tokenizer_path() -> str:
	models_dir = (SCRIPT_DIR.parent / "models").resolve()
	if not models_dir.exists():
		return ""

	preferred_names = (
		"mom-embedding-pro",
		"mom-embedding-light",
		"mom-embedding-ultra",
	)

	candidates: List[Path] = []
	for name in preferred_names:
		path = models_dir / name
		if (path / "tokenizer.json").exists() and (path / "tokenizer_config.json").exists():
			candidates.append(path)

	if not candidates:
		for tokenizer_json in models_dir.rglob("tokenizer.json"):
			if "onnx" in tokenizer_json.parts:
				continue
			parent = tokenizer_json.parent
			if (parent / "tokenizer_config.json").exists():
				candidates.append(parent)

	if not candidates:
		return ""

	# Deterministic selection: shortest path string then lexicographic.
	candidates = sorted(candidates, key=lambda p: (len(str(p)), str(p)))
	return str(candidates[0])


def normalize_text(value: Any) -> str:
	text = str(value or "")
	return " ".join(text.replace("\r", "\n").split()).strip()


def first_text(sample: Dict[str, Any], keys: Sequence[str]) -> str:
	for key in keys:
		value = sample.get(key)
		if isinstance(value, str):
			text = normalize_text(value)
			if text:
				return text
	return ""


def message_role(message: Dict[str, Any]) -> str:
	for key in ("role", "from", "speaker", "author", "name"):
		value = message.get(key)
		if value is not None:
			return normalize_text(value).lower()
	return ""


def message_content(message: Dict[str, Any]) -> str:
	for key in ("content", "text", "value", "message", "utterance", "body"):
		value = message.get(key)
		if isinstance(value, str):
			text = normalize_text(value)
			if text:
				return text
	return ""


def iter_message_lists(value: Any) -> Iterator[List[Dict[str, Any]]]:
	if isinstance(value, (list, tuple)) and value and all(isinstance(item, dict) for item in value):
		yield list(value)
		return
	if hasattr(value, "tolist"):
		try:
			as_list = value.tolist()
			if isinstance(as_list, list) and as_list and all(isinstance(item, dict) for item in as_list):
				yield as_list
				return
		except Exception:
			pass
	if isinstance(value, dict):
		for key in ("messages", "conversation", "dialogue", "turns"):
			nested = value.get(key)
			if isinstance(nested, (list, tuple)) and nested and all(isinstance(item, dict) for item in nested):
				yield list(nested)
			elif hasattr(nested, "tolist"):
				try:
					as_list = nested.tolist()
					if isinstance(as_list, list) and as_list and all(isinstance(item, dict) for item in as_list):
						yield as_list
				except Exception:
					pass


def extract_from_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
	user_parts: List[str] = []
	assistant_parts: List[str] = []
	ordered_parts: List[str] = []
	for message in messages:
		content = message_content(message)
		if not content:
			continue
		ordered_parts.append(content)
		role = message_role(message)
		if role in ROLE_ASSISTANT:
			assistant_parts.append(content)
		elif role in ROLE_USER:
			user_parts.append(content)
		elif not user_parts:
			user_parts.append(content)
		else:
			assistant_parts.append(content)
	request = "\n".join(user_parts).strip()
	response = "\n".join(assistant_parts).strip()
	if not response and len(ordered_parts) >= 2:
		# Some public chat datasets don't annotate assistant role in generation split.
		request = request or "\n".join(ordered_parts[:-1]).strip()
		response = ordered_parts[-1].strip()
	if not request and messages:
		request = message_content(messages[0])
	if not response and len(messages) > 1:
		response = message_content(messages[-1])
	return normalize_text(request), normalize_text(response)


class TokenCounter:
	def __init__(self, tokenizer_name: str) -> None:
		self.tokenizer = None
		self.backend = "regex"
		try:
			transformers = importlib.import_module("transformers")
			# Avoid network retries in restricted environments; use local cache only.
			self.tokenizer = transformers.AutoTokenizer.from_pretrained(
				tokenizer_name,
				use_fast=True,
				local_files_only=True,
			)
			self.backend = f"transformers:{tokenizer_name}"
		except Exception:
			self.tokenizer = None
			self.backend = "regex"

	def count(self, text: str) -> int:
		if self.tokenizer is not None:
			try:
				return len(self.tokenizer.encode(text, add_special_tokens=False))
			except Exception:
				pass
		return len(TOKEN_SPLIT_RE.findall(text))


def classify_output_length(token_count: int, easy_max: int, hard_min: int) -> str:
	if token_count >= hard_min:
		return "hard"
	if token_count <= easy_max:
		return "easy"
	return "medium"


def extract_request_and_response(sample: Dict[str, Any]) -> Tuple[str, str]:
	request = first_text(sample, REQUEST_KEYS)
	response = first_text(sample, RESPONSE_KEYS)

	if request and response:
		return request, response

	conversation_candidates: List[Tuple[str, str]] = []
	for key in CONVERSATION_KEYS:
		value = sample.get(key)
		for messages in iter_message_lists(value):
			request_text, response_text = extract_from_messages(messages)
			if request_text and response_text:
				conversation_candidates.append((request_text, response_text))
			elif request_text and not request:
				request = request_text
			elif response_text and not response:
				response = response_text

	if conversation_candidates:
		if not request:
			request = conversation_candidates[0][0]
		if not response:
			response = max(conversation_candidates, key=lambda pair: len(pair[1]))[1]

	if not request:
		request = first_text(sample, ("question", "prompt", "instruction", "input", "text"))
	if not response:
		response = first_text(sample, ("answer", "completion", "output", "chosen", "response"))

	return normalize_text(request), normalize_text(response)


def load_with_datasets(dataset_name: str, split: str, seed: int) -> Iterable[Dict[str, Any]]:
	load_dataset = importlib.import_module("datasets").load_dataset

	try:
		dataset = load_dataset(dataset_name, split=split, streaming=True)
		dataset = dataset.shuffle(seed=seed, buffer_size=10000)
		for item in dataset:
			yield dict(item)
		return
	except Exception:
		pass

	dataset = load_dataset(dataset_name, split=split)
	dataset = dataset.shuffle(seed=seed)
	for item in dataset:
		yield dict(item)


def load_with_hf_hub(dataset_name: str, split: str, seed: int, hf_endpoint: str) -> Iterable[Dict[str, Any]]:
	snapshot_download = importlib.import_module("huggingface_hub").snapshot_download

	local_dir = snapshot_download(
		repo_id=dataset_name,
		repo_type="dataset",
		endpoint=hf_endpoint or None,
	)
	root = Path(local_dir)
	files = sorted(root.rglob("*"))
	data_files = [path for path in files if path.suffix.lower() in {".parquet", ".jsonl", ".json", ".csv"}]
	if not data_files:
		raise RuntimeError(f"No supported data files found in dataset snapshot: {dataset_name}")

	split_matches = [path for path in data_files if split.lower() in path.name.lower()]
	target_files = split_matches if split_matches else data_files

	rows: List[Dict[str, Any]] = []
	for path in target_files:
		if path.suffix.lower() == ".parquet":
			pd = importlib.import_module("pandas")

			rows.extend(pd.read_parquet(path).to_dict(orient="records"))
		elif path.suffix.lower() in {".jsonl", ".json"}:
			with path.open("r", encoding="utf-8") as handle:
				if path.suffix.lower() == ".jsonl":
					for line in handle:
						line = line.strip()
						if line:
							rows.append(json.loads(line))
				else:
					loaded = json.load(handle)
					if isinstance(loaded, list):
						rows.extend(item for item in loaded if isinstance(item, dict))
					elif isinstance(loaded, dict):
						rows.append(loaded)
		elif path.suffix.lower() == ".csv":
			pd = importlib.import_module("pandas")

			rows.extend(pd.read_csv(path).to_dict(orient="records"))

	random.Random(seed).shuffle(rows)
	for item in rows:
		yield dict(item)


def parse_fallback_datasets(value: str) -> List[str]:
	items = [item.strip() for item in value.split(",")]
	return [item for item in items if item]


def resolve_split_for_dataset(dataset_name: str, requested_split: str) -> str:
	if dataset_name == "HuggingFaceH4/ultrachat_200k" and requested_split == "train":
		return "train_sft"
	return requested_split


def iter_samples(
	dataset_name: str,
	split: str,
	seed: int,
	hf_endpoint: str,
	fallback_datasets: Sequence[str],
) -> Iterable[Dict[str, Any]]:
	errors: List[str] = []
	candidates = [dataset_name, *fallback_datasets]

	for candidate in candidates:
		effective_split = resolve_split_for_dataset(candidate, split)
		try:
			print(f"[info] trying datasets loader for {candidate} split={effective_split}")
			for item in load_with_datasets(candidate, effective_split, seed):
				yield item
			return
		except Exception as exc:
			msg = f"datasets loader failed for {candidate}: {exc}"
			errors.append(msg)
			print(f"[warn] {msg}")

		try:
			print(f"[info] trying snapshot fallback for {candidate} split={effective_split}")
			for item in load_with_hf_hub(candidate, effective_split, seed, hf_endpoint):
				yield item
			return
		except Exception as exc:
			msg = f"snapshot fallback failed for {candidate}: {exc}"
			errors.append(msg)
			print(f"[warn] {msg}")

	error_text = " | ".join(errors) if errors else "unknown error"
	raise RuntimeError(
		"Failed to load dataset from primary and fallback sources. "
		f"Details: {error_text}. "
		"If you must use lmsys/chatbot_arena_conversations, request access on Hugging Face first."
	)


def build_records(
	samples: Iterable[Dict[str, Any]],
	token_counter: TokenCounter,
	target_count: int,
	easy_max: int,
	hard_min: int,
) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	for sample in samples:
		request, response = extract_request_and_response(sample)
		if not request or not response:
			continue
		token_count = token_counter.count(response)
		label = classify_output_length(token_count, easy_max, hard_min)
		records.append({"request": request, "label": label})
		if len(records) >= target_count:
			break
	return records


def split_records(records: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	if not records:
		raise ValueError("no records to split")
	random.Random(seed).shuffle(records)
	train_count = int(len(records) * train_ratio)
	train_count = max(1, min(train_count, len(records) - 1))
	return records[:train_count], records[train_count:]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, ensure_ascii=False))
			handle.write("\n")


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
	return dict(Counter(row["label"] for row in records))


def main() -> None:
	args = parse_args()
	if args.train_ratio <= 0 or args.train_ratio >= 1:
		raise SystemExit("--train-ratio must be between 0 and 1")
	if args.sample_size < 2:
		raise SystemExit("--sample-size must be at least 2")
	if args.easy_max < 0:
		raise SystemExit("--easy-max must be >= 0")
	if args.hard_min <= args.easy_max:
		raise SystemExit("--hard-min must be greater than --easy-max")

	if args.hf_endpoint:
		os.environ["HF_ENDPOINT"] = args.hf_endpoint
	fallback_datasets = parse_fallback_datasets(args.fallback_datasets)

	output_dir = resolve_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	train_path = output_dir / args.train_file
	valid_path = output_dir / args.valid_file
	selected_tokenizer = args.tokenizer
	if args.tokenizer == "auto":
		local_tokenizer = find_local_tokenizer_path()
		selected_tokenizer = local_tokenizer or "gpt2"

	print(f"[info] dataset={args.dataset} split={args.split} target_samples={args.sample_size}")
	print(f"[info] hf_endpoint={os.environ.get('HF_ENDPOINT', '')}")
	print(f"[info] fallback_datasets={fallback_datasets}")
	print(f"[info] tokenizer={selected_tokenizer}")
	print(
		f"[info] thresholds: easy<= {args.easy_max}, medium=({args.easy_max + 1}..{args.hard_min - 1}), hard>= {args.hard_min}"
	)

	samples = iter_samples(
		args.dataset,
		args.split,
		args.seed,
		args.hf_endpoint,
		fallback_datasets,
	)
	token_counter = TokenCounter(selected_tokenizer)
	print(f"[info] token_counter_backend={token_counter.backend}")
	records = build_records(
		samples,
		token_counter,
		args.sample_size,
		args.easy_max,
		args.hard_min,
	)

	if len(records) < args.sample_size:
		raise SystemExit(f"not enough valid samples: got {len(records)}, expected {args.sample_size}")

	train_rows, valid_rows = split_records(records, args.train_ratio, args.seed)
	write_jsonl(train_path, train_rows)
	write_jsonl(valid_path, valid_rows)

	label_counts = summarize(records)
	print(f"[ok] total={len(records)} train={len(train_rows)} valid={len(valid_rows)}")
	print(f"[ok] labels={label_counts}")
	print(f"[ok] train_file={train_path}")
	print(f"[ok] valid_file={valid_path}")


if __name__ == "__main__":
	main()
