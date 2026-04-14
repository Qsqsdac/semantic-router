#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = SCRIPT_DIR / ".build" / "complexity" / "train.jsonl"
DEFAULT_KEYWORD_MAP = SCRIPT_DIR / "keyword_map.json"

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']*")
STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract easy/hard discriminative keywords and update keyword_map.json "
            "with probability constraints against medium and opposite class"
        )
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_TRAIN_FILE),
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--keyword-map",
        default=str(DEFAULT_KEYWORD_MAP),
        help="Path to keyword_map.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top K keywords per label to add",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=8,
        help="Minimum document frequency in label to be considered",
    )
    parser.add_argument(
        "--min-target-prob",
        type=float,
        default=0.015,
        help="Minimum document probability in target label",
    )
    parser.add_argument(
        "--max-other-prob",
        type=float,
        default=0.007,
        help="Maximum document probability allowed in each non-target label",
    )
    parser.add_argument(
        "--max-ngrams",
        type=int,
        default=2,
        choices=[1, 2],
        help="Use unigram only (1) or unigram+bigram (2)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing easy/hard keywords instead of merge",
    )
    return parser.parse_args()


def normalize_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def tokenize(text: str) -> List[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if len(token) >= 3 and token not in STOPWORDS]


def extract_features(tokens: Sequence[str], max_ngrams: int) -> Set[str]:
    features: Set[str] = set(tokens)
    if max_ngrams >= 2 and len(tokens) >= 2:
        for idx in range(len(tokens) - 1):
            features.add(f"{tokens[idx]} {tokens[idx + 1]}")
    return features


def iter_train_rows(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid json at {path}:{line_no} ({exc})")

            if not isinstance(obj, dict):
                continue
            text = str(obj.get("request", "") or "").strip()
            label = str(obj.get("label", "") or "").strip().lower()
            if not text or label not in {"easy", "medium", "hard"}:
                continue
            yield text, label


def rank_keywords(
    target_df: Counter,
    other_a_df: Counter,
    other_b_df: Counter,
    target_docs: int,
    other_a_docs: int,
    other_b_docs: int,
    min_df: int,
    min_target_prob: float,
    max_other_prob: float,
    top_k: int,
) -> List[str]:
    candidates: List[Tuple[float, int, str]] = []
    eps = 1e-12

    def prob(df: int, docs: int) -> float:
        if docs <= 0:
            return 0.0
        return float(df) / float(docs)

    for token, df in target_df.items():
        if df < min_df:
            continue

        p_target = prob(df, target_docs)
        p_other_a = prob(other_a_df.get(token, 0), other_a_docs)
        p_other_b = prob(other_b_df.get(token, 0), other_b_docs)

        if p_target < min_target_prob:
            continue
        if p_other_a > max_other_prob or p_other_b > max_other_prob:
            continue

        other_max = max(p_other_a, p_other_b)
        # Score favors high target prevalence and low prevalence in both non-target labels.
        score = p_target * math.log((p_target + eps) / (other_max + eps))
        if score <= 0:
            continue
        candidates.append((score, df, token))

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [token for _, _, token in candidates[:top_k]]


def merge_keywords(existing: List[str], discovered: List[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()

    for token in existing + discovered:
        cleaned = str(token or "").strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def load_keyword_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {"hard": [], "easy": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"hard": [], "easy": []}
    return {
        "hard": list(data.get("hard", []) or []),
        "easy": list(data.get("easy", []) or []),
    }


def main() -> None:
    args = parse_args()
    train_file = normalize_path(args.train_file)
    keyword_map = normalize_path(args.keyword_map)

    if not train_file.exists():
        raise SystemExit(f"train file not found: {train_file}")

    easy_docs = 0
    medium_docs = 0
    hard_docs = 0
    easy_df: Counter = Counter()
    medium_df: Counter = Counter()
    hard_df: Counter = Counter()

    for text, label in iter_train_rows(train_file):
        tokens = tokenize(text)
        if not tokens:
            continue
        features = extract_features(tokens, args.max_ngrams)
        if label == "easy":
            easy_docs += 1
            easy_df.update(features)
        elif label == "medium":
            medium_docs += 1
            medium_df.update(features)
        elif label == "hard":
            hard_docs += 1
            hard_df.update(features)

    if easy_docs == 0 or medium_docs == 0 or hard_docs == 0:
        raise SystemExit("train file must contain easy/medium/hard samples")

    discovered_easy = rank_keywords(
        target_df=easy_df,
        other_a_df=hard_df,
        other_b_df=medium_df,
        target_docs=easy_docs,
        other_a_docs=hard_docs,
        other_b_docs=medium_docs,
        min_df=args.min_df,
        min_target_prob=args.min_target_prob,
        max_other_prob=args.max_other_prob,
        top_k=args.top_k,
    )
    discovered_hard = rank_keywords(
        target_df=hard_df,
        other_a_df=easy_df,
        other_b_df=medium_df,
        target_docs=hard_docs,
        other_a_docs=easy_docs,
        other_b_docs=medium_docs,
        min_df=args.min_df,
        min_target_prob=args.min_target_prob,
        max_other_prob=args.max_other_prob,
        top_k=args.top_k,
    )

    current_map = load_keyword_map(keyword_map)
    if args.replace:
        updated_easy = discovered_easy
        updated_hard = discovered_hard
    else:
        updated_easy = merge_keywords(current_map["easy"], discovered_easy)
        updated_hard = merge_keywords(current_map["hard"], discovered_hard)

    updated = {
        "hard": updated_hard,
        "easy": updated_easy,
    }

    keyword_map.parent.mkdir(parents=True, exist_ok=True)
    keyword_map.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[info] keyword map updated")
    print(f"  train_file: {train_file}")
    print(f"  keyword_map: {keyword_map}")
    print(
        f"  easy_docs: {easy_docs}, medium_docs: {medium_docs}, hard_docs: {hard_docs}"
    )
    print(
        f"  constraints: min_target_prob={args.min_target_prob}, max_other_prob={args.max_other_prob}, min_df={args.min_df}"
    )
    print(
        "  discovered_easy:",
        ", ".join(discovered_easy[:10]) if discovered_easy else "(none)",
    )
    print(
        "  discovered_hard:",
        ", ".join(discovered_hard[:10]) if discovered_hard else "(none)",
    )
    print(f"  easy_keywords_total: {len(updated_easy)}")
    print(f"  hard_keywords_total: {len(updated_hard)}")


if __name__ == "__main__":
    main()