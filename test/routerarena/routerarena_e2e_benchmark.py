#!/usr/bin/env python3
"""RouterArena-backed end-to-end benchmark for semantic-router.

The benchmark only talks to the public OpenAI-compatible chat completions API
documented in src/vllm-sr/tutorials.md.

It evaluates two things:
1. Task accuracy using RouterArena-style intrinsic scoring for each dataset.
2. Router robustness by comparing the selected model between the full and
   robustness splits.

The routing latency is read from the x-vsr-total-routing-latency-ms header.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTERARENA_ROOT = Path("/home/chengsixiang/RouterArena")
ROUTERARENA_CONFIG_DIR = ROUTERARENA_ROOT / "config" / "eval_config" / "zero-shot"
DEFAULT_DATASET = "RouteWorks/RouterArena"
DEFAULT_ENDPOINT = "/v1/chat/completions"
LATENCY_HEADER = "x-vsr-total-routing-latency-ms"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "routerarena-e2e"
DEFAULT_SPLITS = ["full", "robustness"]
DEFAULT_AUTH_TOKEN = "sk-123456"
DEFAULT_USER_ID = "demo-user"
DEFAULT_USER_GROUPS = "premium-tier"

SUPPORTED_SPLITS = {"sub_10", "full", "robustness"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RouterArena-backed end-to-end benchmark for /v1/chat/completions"
    )
    parser.add_argument(
        "--router-url",
        default="http://localhost:9099",
        help="OpenAI-compatible router URL",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Chat completions endpoint path",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=sorted(SUPPORTED_SPLITS),
        help="RouterArena splits to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples per split. 0 means no limit.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel HTTP workers",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where detail and summary files are written",
    )
    parser.add_argument(
        "--auth-token",
        default=DEFAULT_AUTH_TOKEN,
        help="Bearer token used in the tutorial example",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="Value for x-authz-user-id",
    )
    parser.add_argument(
        "--user-groups",
        default=DEFAULT_USER_GROUPS,
        help="Value for x-authz-user-groups",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help="Value for reasoning_effort in the request body",
    )
    parser.add_argument(
        "--model",
        default="MoM",
        help="OpenAI-compatible model field used in the tutorial example",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_answer(value: Any) -> str:
    text = normalize_text(value)
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_chess_move(move: Any) -> str:
    text = normalize_text(move)
    if not text:
        return ""
    if len(text) == 1 and text.isalpha():
        return text
    if text.isdigit():
        value = int(text)
        if 0 <= value <= 25:
            return chr(ord("a") + value)
        return text
    text = text.replace("-", "").replace(" ", "").replace("_", "")
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def safe_format_prompt(template: str, **kwargs: Any) -> str:
    escaped: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            escaped[key] = value.replace("{", "{{").replace("}", "}}")
        else:
            escaped[key] = value
    return template.format(**escaped)


def options_to_string(options: Any) -> str:
    if options is None:
        return ""
    if not isinstance(options, list):
        try:
            options = list(options)
        except Exception:
            options = []

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for idx, option in enumerate(options):
        letter = letters[idx] if idx < len(letters) else str(idx)
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def load_eval_configs() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for path in sorted(ROUTERARENA_CONFIG_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        configs[cfg["eval_params"]["dataset"]] = cfg["eval_params"]
    return configs


def read_parquet_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.glob(f"{split}*.parquet"))
    if not parquet_files:
        parquet_files = sorted(dataset_dir.rglob(f"{split}*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split: {split}")

    frames = [pd.read_parquet(path) for path in parquet_files]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def load_routerarena_split(dataset_repo: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
    local_dir = Path(snapshot_download(repo_id=dataset_repo, repo_type="dataset"))
    frame = read_parquet_split(local_dir, split)
    if max_samples > 0:
        frame = frame.head(max_samples)
    return frame.to_dict(orient="records")


def resolve_dataset_name(row: Dict[str, Any]) -> str:
    dataset_name = row.get("Dataset name")
    if dataset_name:
        return str(dataset_name)

    global_index = str(row.get("Global Index") or row.get("global index") or "")
    parts = global_index.split("_")
    if not parts:
        return ""

    if parts[0] == "Ethics" and len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    if parts[0] == "ChessInstruct" and len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def base_dataset_name(row: Dict[str, Any]) -> str:
    dataset_name = resolve_dataset_name(row)
    options = row.get("Options")
    if options is None:
        has_options = False
    elif hasattr(options, "__len__"):
        has_options = len(options) > 0
    else:
        has_options = bool(options)

    if "Ethics" in dataset_name:
        return dataset_name
    if "ChessInstruct" in dataset_name:
        return "ChessInstruct_mcq" if has_options else "ChessInstruct"
    return str(dataset_name).split("_", 1)[0]


def build_prompt(row: Dict[str, Any], eval_params: Dict[str, Any]) -> str:
    question = str(row.get("Question", "")).strip()
    context_value = row.get("Context", "")
    context = str(context_value) if context_value not in [None, ""] else "None"
    options = options_to_string(row.get("Options"))
    answer = ""

    if "is_stdin_prompt" in eval_params or "not_is_stdin_prompt" in eval_params:
        template = eval_params.get("not_is_stdin_prompt") or "{Question}"
        return safe_format_prompt(template, Question=question)

    template = eval_params.get("prompt", "{Question}")
    return safe_format_prompt(
        template,
        Question=question,
        Context=context,
        Options=options,
        Answer=answer,
    )


def extract_response_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", []) if isinstance(payload, dict) else []
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return str(content)
            content = first.get("text")
            if content is not None:
                return str(content)
    return ""


def extract_boxed_answer(text: str) -> str:
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match[-1].strip()

    match = re.findall(r"\b([A-Z])\b", text)
    if match:
        return match[-1].strip()

    return text.strip()


def parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = normalize_text(value)
    text = text.replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except Exception:
        return None


def math_equal(prediction: str, reference: str, **kwargs: Any) -> bool:
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    if pred_norm == ref_norm:
        return True

    pred_num = parse_numeric(prediction)
    ref_num = parse_numeric(reference)
    if pred_num is not None and ref_num is not None:
        if abs(pred_num - ref_num) <= 1e-9:
            return True
        if abs(pred_num - ref_num / 100.0) <= 1e-9:
            return True
        if abs(pred_num * 100.0 - ref_num) <= 1e-9:
            return True

    if "=" in prediction and "=" in reference:
        return normalize_answer(prediction.split("=")[-1]) == normalize_answer(reference.split("=")[-1])

    return False


def convert_ground_truth_for_mcq(ground_truth: Any, options: Any) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if isinstance(ground_truth, str):
        stripped = ground_truth.strip()
        if stripped.upper() in letters:
            return stripped.upper()
        if stripped.isdigit():
            index = int(stripped)
            if 0 <= index < len(letters):
                return letters[index]
        try:
            index = int(float(stripped))
            if 0 <= index < len(letters):
                return letters[index]
        except Exception:
            pass

    if isinstance(options, list):
        normalized_ground_truth = normalize_answer(ground_truth)
        for idx, option in enumerate(options):
            if normalize_answer(option) == normalized_ground_truth:
                return letters[idx]

    return normalize_answer(ground_truth)


def mcq_score(prediction_text: str, ground_truth: Any, options: Any = None, **kwargs: Any) -> float:
    extracted = extract_boxed_answer(prediction_text)
    expected = convert_ground_truth_for_mcq(ground_truth, options)

    if expected in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(expected) == 1:
        if extracted.strip().upper() == expected:
            return 1.0

    normalized_pred = normalize_answer(extracted)
    normalized_gt = normalize_answer(ground_truth)
    if normalized_pred == normalized_gt:
        return 1.0

    if isinstance(options, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx, option in enumerate(options):
            if normalize_answer(option) == normalized_pred and letters[idx] == expected:
                return 1.0

    return 0.0


def exact_match_score(prediction_text: str, ground_truth: Any, **kwargs: Any) -> float:
    extracted = extract_boxed_answer(prediction_text)
    return 1.0 if normalize_answer(extracted) == normalize_answer(ground_truth) else 0.0


def meteor_score(prediction_text: str, ground_truth: Any, **kwargs: Any) -> float:
    pred = normalize_answer(extract_boxed_answer(prediction_text))
    ref = normalize_answer(ground_truth)
    if not pred or not ref:
        return 0.0

    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    matches = sum(min(pred_counter[token], ref_counter[token]) for token in pred_counter if token in ref_counter)
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    if precision == 0.0 and recall == 0.0:
        return 0.0

    fmean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) > 0 else 0.0
    penalty = 0.0
    return max(0.0, min(1.0, fmean * (1.0 - penalty)))


def chess_score(prediction_text: str, ground_truth: Any, **kwargs: Any) -> float:
    extracted = extract_boxed_answer(prediction_text)
    return 1.0 if normalize_chess_move(extracted) == normalize_chess_move(ground_truth) else 0.0


def superglue_exact_match_score(prediction_text: str, ground_truth: Any, **kwargs: Any) -> float:
    extracted = extract_boxed_answer(prediction_text)
    gt = ground_truth

    if isinstance(gt, str):
        stripped = gt.strip()
        if stripped.lower() in {"yes", "no"}:
            mapping = {
                "a": "yes",
                "x": "yes",
                "1": "yes",
                "true": "yes",
                "b": "no",
                "y": "no",
                "0": "no",
                "false": "no",
            }
            extracted_norm = normalize_text(extracted)
            mapped = mapping.get(extracted_norm, extracted_norm)
            return 1.0 if mapped == stripped.lower() else 0.0

        if stripped in {"0", "1", "0.0", "1.0"}:
            mapping = {
                "a": "0.0",
                "0": "0.0",
                "false": "0.0",
                "b": "1.0",
                "1": "1.0",
                "true": "1.0",
            }
            extracted_norm = normalize_text(extracted)
            mapped = mapping.get(extracted_norm, extracted_norm)
            return 1.0 if mapped == stripped else 0.0

    return 1.0 if normalize_answer(extracted) == normalize_answer(gt) else 0.0


def superglue_cloze_score(prediction_text: str, ground_truth: Any, **kwargs: Any) -> float:
    extracted = extract_boxed_answer(prediction_text)
    if isinstance(ground_truth, int):
        expected_letter = chr(ord("A") + ground_truth)
        return 1.0 if extracted.strip().upper() == expected_letter else 0.0

    if isinstance(ground_truth, str):
        stripped = ground_truth.strip()
        if len(stripped) == 1 and stripped.upper() in string.ascii_uppercase:
            return 1.0 if extracted.strip().upper() == stripped.upper() else 0.0
        try:
            idx = int(float(stripped))
            if 0 <= idx < 26:
                expected_letter = chr(ord("A") + idx)
                return 1.0 if extracted.strip().upper() == expected_letter else 0.0
        except Exception:
            pass

    return 1.0 if normalize_answer(extracted) == normalize_answer(ground_truth) else 0.0


def code_score(_: str, __: Any, **kwargs: Any) -> Optional[float]:
    return None


METRIC_FUNCS = {
    "mcq_accuracy": mcq_score,
    "mcq_exact_match": mcq_score,
    "math_metric": math_equal,
    "exact_match": exact_match_score,
    "meteor_score": meteor_score,
    "chess_accuracy": chess_score,
    "superglue_exact_match": superglue_exact_match_score,
    "superglue_clozetest": superglue_cloze_score,
    "code_accuracy": code_score,
}


EVAL_CONFIGS = load_eval_configs()


def resolve_metric_name(dataset_name: str, eval_params: Dict[str, Any]) -> str:
    metrics = eval_params.get("eval_metrics") or []
    if metrics:
        return str(metrics[0])
    fallback = {
        "MMLUPro": "mcq_accuracy",
        "LiveCodeBench": "code_accuracy",
        "NarrativeQA": "meteor_score",
    }
    return fallback.get(dataset_name, "mcq_accuracy")


def classify_dataset(row: Dict[str, Any]) -> Tuple[str, str]:
    dataset_name = base_dataset_name(row)
    eval_params = EVAL_CONFIGS[dataset_name]
    metric_name = resolve_metric_name(dataset_name, eval_params)
    return dataset_name, metric_name


def build_sample_prompt(row: Dict[str, Any]) -> Tuple[str, str, str]:
    dataset_name, metric_name = classify_dataset(row)
    eval_params = EVAL_CONFIGS[dataset_name]
    prompt = build_prompt(row, eval_params)
    return dataset_name, metric_name, prompt


def call_chat_completion(
    router_url: str,
    endpoint: str,
    model: str,
    prompt: str,
    auth_token: str,
    user_id: str,
    user_groups: str,
    reasoning_effort: str,
    timeout: int,
) -> Tuple[requests.Response, float]:
    url = f"{router_url.rstrip('/')}{endpoint}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": reasoning_effort,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
        "x-authz-user-id": user_id,
        "x-authz-user-groups": user_groups,
    }

    started = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    http_elapsed_ms = (time.perf_counter() - started) * 1000.0
    return response, http_elapsed_ms


def parse_response_metadata(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {}

    response_text = extract_response_text(payload)
    selected_model = None
    if isinstance(payload, dict):
        selected_model = payload.get("model") or payload.get("selected_model")
    if not selected_model:
        selected_model = response.headers.get("x-selected-model") or response.headers.get("x-vsr-selected-model")

    latency_raw = response.headers.get(LATENCY_HEADER)
    try:
        routing_latency_ms = float(latency_raw) if latency_raw is not None else None
    except (TypeError, ValueError):
        routing_latency_ms = None

    return {
        "raw_response": payload,
        "response_text": response_text,
        "selected_model": selected_model,
        "routing_latency_ms": routing_latency_ms,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, set):
        return list(value)
    return str(value)


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_one(
    index: int,
    row: Dict[str, Any],
    router_url: str,
    endpoint: str,
    model: str,
    auth_token: str,
    user_id: str,
    user_groups: str,
    reasoning_effort: str,
    timeout: int,
) -> Dict[str, Any]:
    dataset_name, metric_name, prompt = build_sample_prompt(row)
    ground_truth = row.get("Answer", "")
    options = row.get("Options")
    global_index = row.get("Global Index") or row.get("global_index") or row.get("global index")

    result: Dict[str, Any] = {
        "index": index,
        "global_index": global_index,
        "dataset_name": dataset_name,
        "metric_name": metric_name,
        "expected_raw": ground_truth,
        "question": str(row.get("Question", "")).strip(),
        "context": str(row.get("Context", "")).strip(),
        "options": options,
        "prompt": prompt,
    }

    started = time.perf_counter()
    try:
        response, http_elapsed_ms = call_chat_completion(
            router_url,
            endpoint,
            model,
            prompt,
            auth_token,
            user_id,
            user_groups,
            reasoning_effort,
            timeout,
        )
        parsed = parse_response_metadata(response)
        task_score = METRIC_FUNCS.get(metric_name, code_score)(
            parsed["response_text"],
            ground_truth,
            options=options,
        )

        result.update(
            {
                "status": "ok" if response.status_code == 200 else "error",
                "http_status": response.status_code,
                "selected_model": parsed["selected_model"],
                "response_text": parsed["response_text"],
                "routing_latency_ms": parsed["routing_latency_ms"],
                "http_elapsed_ms": http_elapsed_ms,
                "task_score": task_score,
                "is_supported": task_score is not None,
                "raw_response": parsed["raw_response"],
            }
        )
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}"
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "http_status": None,
                "selected_model": None,
                "response_text": "",
                "routing_latency_ms": None,
                "http_elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "task_score": None,
                "is_supported": False,
                "raw_response": None,
                "error": str(exc),
            }
        )

    return result


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    fail_rows = [row for row in rows if row.get("status") != "ok"]
    scored_rows = [row for row in ok_rows if isinstance(row.get("task_score"), (int, float))]
    unsupported_rows = [row for row in ok_rows if row.get("task_score") is None]

    total_score = sum(float(row["task_score"]) for row in scored_rows)
    avg_score = total_score / len(scored_rows) if scored_rows else None

    routing_latencies = [float(row["routing_latency_ms"]) for row in ok_rows if isinstance(row.get("routing_latency_ms"), (int, float))]
    http_latencies = [float(row["http_elapsed_ms"]) for row in ok_rows if isinstance(row.get("http_elapsed_ms"), (int, float))]

    dataset_counter = Counter(row.get("dataset_name") for row in rows if row.get("dataset_name"))
    metric_counter = Counter(row.get("metric_name") for row in rows if row.get("metric_name"))

    return {
        "total_samples": len(rows),
        "ok_count": len(ok_rows),
        "fail_count": len(fail_rows),
        "scored_count": len(scored_rows),
        "unsupported_count": len(unsupported_rows),
        "accuracy": avg_score,
        "avg_routing_latency_ms": (sum(routing_latencies) / len(routing_latencies)) if routing_latencies else None,
        "p50_routing_latency_ms": percentile(routing_latencies, 50),
        "p95_routing_latency_ms": percentile(routing_latencies, 95),
        "p99_routing_latency_ms": percentile(routing_latencies, 99),
        "avg_http_elapsed_ms": (sum(http_latencies) / len(http_latencies)) if http_latencies else None,
        "dataset_distribution": dict(dataset_counter),
        "metric_distribution": dict(metric_counter),
    }


def compute_robustness(full_rows: List[Dict[str, Any]], robust_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    full_map = {
        str(row.get("global_index")): row
        for row in full_rows
        if row.get("status") == "ok" and row.get("selected_model") and row.get("global_index") is not None
    }
    robust_map = {
        str(row.get("global_index")): row
        for row in robust_rows
        if row.get("status") == "ok" and row.get("selected_model") and row.get("global_index") is not None
    }

    common_indices = sorted(set(full_map).intersection(robust_map))
    if not common_indices:
        return {
            "matched_samples": 0,
            "flips": 0,
            "flip_rate": None,
            "stability": None,
        }

    flips = 0
    for key in common_indices:
        if normalize_text(full_map[key]["selected_model"]) != normalize_text(robust_map[key]["selected_model"]):
            flips += 1

    matched_samples = len(common_indices)
    flip_rate = flips / matched_samples
    return {
        "matched_samples": matched_samples,
        "flips": flips,
        "flip_rate": flip_rate,
        "stability": 1.0 - flip_rate,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split in args.splits:
        print(f"[info] loading RouterArena split: {split}")
        samples = load_routerarena_split(args.dataset, split, args.max_samples)
        print(f"[info] split={split}, samples={len(samples)}")

        start = time.time()
        rows: List[Optional[Dict[str, Any]]] = [None] * len(samples)
        worker_count = max(1, int(args.workers))

        if worker_count == 1:
            for index, sample in enumerate(samples, start=1):
                rows[index - 1] = evaluate_one(
                    index,
                    sample,
                    args.router_url,
                    args.endpoint,
                    args.model,
                    args.auth_token,
                    args.user_id,
                    args.user_groups,
                    args.reasoning_effort,
                    args.timeout,
                )
                if index % 100 == 0:
                    print(f"[progress] {split}: {index}/{len(samples)}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        evaluate_one,
                        index,
                        sample,
                        args.router_url,
                        args.endpoint,
                        args.model,
                        args.auth_token,
                        args.user_id,
                        args.user_groups,
                        args.reasoning_effort,
                        args.timeout,
                    ): index
                    for index, sample in enumerate(samples, start=1)
                }
                completed = 0
                for future in concurrent.futures.as_completed(future_map):
                    index = future_map[future]
                    rows[index - 1] = future.result()
                    completed += 1
                    if completed % 100 == 0:
                        print(f"[progress] {split}: {completed}/{len(samples)}")

        split_detail_rows = [row for row in rows if row is not None]
        split_detail_rows.sort(key=lambda item: item["index"])
        split_rows[split] = split_detail_rows

        detail_file = output_dir / f"routerarena_e2e_{split}_detail_{run_id}.jsonl"
        latest_detail_file = output_dir / f"latest_{split}_detail.jsonl"
        write_jsonl(detail_file, split_detail_rows)
        write_jsonl(latest_detail_file, split_detail_rows)

        summary = summarize_rows(split_detail_rows)
        summary.update(
            {
                "run_id": run_id,
                "router_url": args.router_url,
                "endpoint": args.endpoint,
                "dataset": args.dataset,
                "split": split,
                "detail_file": str(detail_file),
                "elapsed_seconds": time.time() - start,
            }
        )
        split_summaries[split] = summary

    robustness_summary = {}
    if "full" in split_rows and "robustness" in split_rows:
        robustness_summary = compute_robustness(split_rows["full"], split_rows["robustness"])

    combined_summary = {
        "run_id": run_id,
        "router_url": args.router_url,
        "endpoint": args.endpoint,
        "dataset": args.dataset,
        "model": args.model,
        "splits": split_summaries,
        "robustness": robustness_summary,
        "note": "RouterArena end-to-end benchmark through /v1/chat/completions. Routing latency comes from x-vsr-total-routing-latency-ms.",
    }

    summary_file = output_dir / f"routerarena_e2e_summary_{run_id}.json"
    latest_summary = output_dir / "latest_summary.json"
    summary_file.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    latest_summary.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    print("[info] benchmark finished")
    print(json.dumps(combined_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()