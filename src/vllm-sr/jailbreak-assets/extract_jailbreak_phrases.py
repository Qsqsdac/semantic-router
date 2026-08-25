#!/usr/bin/env python3
"""Extract discriminative jailbreak phrases from evaluation JSONL data.

The input is the detail JSONL produced by eval_jailbreak_api.py. Phrases are
scored by unsafe frequency and safe frequency, then written as a JSON list that
can be consumed by the MinHash detector.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "because", "by", "can", "do", "for", "from", "get", "how",
    "i", "if", "in", "into", "is", "it", "me", "my", "no", "not", "of",
    "on", "or", "our", "please", "that", "the", "their", "this", "to", "use",
    "want", "what", "with", "you", "your",
}
JAILBREAK_HINTS = {
    "answer", "character", "chatgpt", "developer", "filter", "forget",
    "ignore", "instruction", "jailbreak", "limit", "mode", "pretend",
    "previous", "prompt", "policy", "refuse", "restriction", "role",
    "rule", "system", "unfiltered", "unrestricted",
}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(words: list[str], minimum: int, maximum: int) -> set[str]:
    phrases: set[str] = set()
    for size in range(minimum, maximum + 1):
        for start in range(len(words) - size + 1):
            phrase_words = words[start : start + size]
            if phrase_words[0] in STOPWORDS and phrase_words[-1] in STOPWORDS:
                continue
            if all(word in STOPWORDS for word in phrase_words):
                continue
            phrases.add(" ".join(phrase_words))
    return phrases


def load_rows(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    unsafe: list[list[str]] = []
    safe: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            words = tokens(str(row.get("text", "")))
            if len(words) < 2:
                continue
            if row.get("expected_norm") == "unsafe":
                unsafe.append(words)
            elif row.get("expected_norm") == "safe":
                safe.append(words)
    return unsafe, safe


def extract(path: Path, minimum: int, maximum: int, min_count: int, limit: int) -> list[str]:
    unsafe_rows, safe_rows = load_rows(path)
    unsafe_counts = Counter(phrase for row in unsafe_rows for phrase in ngrams(row, minimum, maximum))
    safe_counts = Counter(phrase for row in safe_rows for phrase in ngrams(row, minimum, maximum))
    unsafe_total = max(len(unsafe_rows), 1)
    safe_total = max(len(safe_rows), 1)

    scored: list[tuple[float, str]] = []
    for phrase, count in unsafe_counts.items():
        if count < min_count:
            continue
        if not JAILBREAK_HINTS.intersection(phrase.split()):
            continue
        unsafe_rate = (count + 0.5) / (unsafe_total + 1)
        safe_rate = (safe_counts[phrase] + 0.5) / (safe_total + 1)
        score = math.log(unsafe_rate / safe_rate) * math.log1p(count)
        scored.append((score, phrase))

    scored.sort(key=lambda item: (-item[0], -len(item[1].split()), item[1]))
    selected: list[str] = []
    for _, phrase in scored:
        # Keep longer phrases when a shorter selected phrase is contained in it.
        if any(phrase in selected_phrase for selected_phrase in selected):
            continue
        selected.append(phrase)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="evaluation detail JSONL")
    parser.add_argument("output", type=Path, help="MinHash pattern JSON file")
    parser.add_argument("--min-n", type=int, default=4)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if args.min_n < 1 or args.max_n < args.min_n:
        parser.error("require 1 <= --min-n <= --max-n")
    patterns = extract(args.input, args.min_n, args.max_n, args.min_count, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(patterns, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(patterns)} patterns to {args.output}")


if __name__ == "__main__":
    main()
