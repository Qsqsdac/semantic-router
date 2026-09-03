#!/usr/bin/env python3
"""OpenAI-compatible record/replay backend for repeatable router experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests


LOGGER = logging.getLogger("openai-record-replay")
WRITE_LOCK = threading.Lock()
CHAT_COMPLETION_PATHS = {"/chat/completions", "/v1/chat/completions"}


def canonical_request(payload: dict[str, Any]) -> bytes:
    """Return a stable representation of every generation-affecting request field."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_request(payload)).hexdigest()


class ReplayStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        path = self.directory / f"{cache_key(payload)}.json"
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read replay record {path}: {exc}") from exc

    def put(self, payload: dict[str, Any], record: dict[str, Any]) -> str:
        key = cache_key(payload)
        target = self.directory / f"{key}.json"
        complete_record = {"request": payload, "recorded_at": int(time.time()), **record}
        with WRITE_LOCK:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.directory, prefix=f".{key}.", delete=False
            ) as handle:
                json.dump(complete_record, handle, ensure_ascii=False, separators=(",", ":"))
                temp_name = handle.name
            Path(temp_name).replace(target)
        return key


def response_record(response: requests.Response, elapsed_ms: float) -> dict[str, Any]:
    return {
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type", "application/json; charset=utf-8"),
        "body": response.content.decode("utf-8", errors="replace"),
        "upstream_elapsed_ms": elapsed_ms,
    }


def new_upstream_session() -> requests.Session:
    """Create an upstream client that does not inherit proxy settings from the environment.

    ``requests`` reads HTTP(S)_PROXY and ALL_PROXY by default.  The replay
    service must be usable with the same direct upstream configuration as the
    router, so proxying is opt-in through ``REPLAY_UPSTREAM_PROXY`` instead.
    """
    session = requests.Session()
    session.trust_env = False
    return session


def import_routerarena_detail(store: ReplayStore, detail_file: Path, reasoning_effort: str) -> tuple[int, int]:
    imported = 0
    skipped = 0
    with detail_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                response = row.get("raw_response")
                model = row.get("selected_model")
                prompt = row.get("prompt")
                if row.get("http_status") != 200 or not isinstance(response, dict) or not model or prompt is None:
                    skipped += 1
                    continue
                payload: dict[str, Any] = {
                    "model": str(model),
                    "messages": [{"role": "user", "content": str(prompt)}],
                }
                if reasoning_effort:
                    payload["reasoning_effort"] = reasoning_effort
                store.put(
                    payload,
                    {
                        "status_code": 200,
                        "content_type": "application/json; charset=utf-8",
                        "body": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                        "upstream_elapsed_ms": row.get("http_elapsed_ms"),
                        "routerarena": {
                            "task_score": row.get("task_score"),
                            "is_supported": row.get("is_supported"),
                            "global_index": row.get("global_index"),
                        },
                    },
                )
                imported += 1
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid detail row {line_number}: {exc}") from exc
    return imported, skipped


class ReplayHandler(BaseHTTPRequestHandler):
    server: "ReplayHTTPServer"
    server_version = "OpenAIRecordReplay/0.1"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _write_json(self, status_code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_record(self, record: dict[str, Any], cache_status: str) -> None:
        body = str(record["body"]).encode("utf-8")
        self.send_response(int(record["status_code"]))
        self.send_header("Content-Type", str(record.get("content_type", "application/json; charset=utf-8")))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-OpenAI-Replay", cache_status)
        if record.get("upstream_elapsed_ms") is not None:
            self.send_header("X-OpenAI-Replay-Original-Elapsed-Ms", str(record["upstream_elapsed_ms"]))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, payload: dict[str, Any]) -> tuple[requests.Response, float]:
        if not self.server.upstream_base:
            raise RuntimeError("REPLAY_UPSTREAM_BASE_URL is required for record or auto cache misses")
        headers = {"Content-Type": "application/json"}
        if self.server.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.server.upstream_api_key}"
        started = time.perf_counter()
        with new_upstream_session() as session:
            response = session.post(
                f"{self.server.upstream_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.server.upstream_timeout,
                proxies=self.server.proxies,
            )
        return response, (time.perf_counter() - started) * 1000.0

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write_json(200, {"status": "ok", "mode": self.server.mode})
            return
        if self.path == "/v1/models":
            self._write_json(200, {"object": "list", "data": []})
            return
        self._write_json(404, {"error": {"message": "not found", "type": "not_found_error"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in CHAT_COMPLETION_PATHS:
            self._write_json(404, {"error": {"message": "not found", "type": "not_found_error"}})
            return
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._write_json(400, {"error": {"message": f"invalid json: {exc}", "type": "invalid_request_error"}})
            return

        try:
            record = self.server.store.get(payload)
            if record is not None and self.server.mode != "record":
                self._write_record(record, "HIT")
                return
            if self.server.mode == "replay":
                self._write_json(404, {"error": {"message": "no recorded response for request and selected model", "type": "replay_miss"}})
                return

            response, elapsed_ms = self._forward(payload)
            record = response_record(response, elapsed_ms)
            record_id = self.server.store.put(payload, record)
            self._write_record(record, f"MISS; id={record_id}")
        except requests.RequestException as exc:
            LOGGER.exception("upstream request failed")
            self._write_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
        except RuntimeError as exc:
            self._write_json(500, {"error": {"message": str(exc), "type": "replay_store_error"}})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


class ReplayHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        store: ReplayStore,
        mode: str,
        upstream_base: str,
        upstream_api_key: str,
        upstream_timeout: int,
        upstream_proxy: str,
    ) -> None:
        super().__init__(address, ReplayHandler)
        self.store = store
        self.mode = mode
        self.upstream_base = upstream_base.rstrip("/")
        self.upstream_api_key = upstream_api_key
        self.upstream_timeout = upstream_timeout
        self.proxies = {"http": upstream_proxy, "https": upstream_proxy} if upstream_proxy else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("REPLAY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("REPLAY_PORT", "18081")))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.getenv("REPLAY_CACHE_DIR", ".cache/openai-replay")))
    parser.add_argument("--mode", choices=("auto", "record", "replay"), default=os.getenv("REPLAY_MODE", "auto"))
    parser.add_argument("--import-routerarena-detail", type=Path)
    parser.add_argument("--reasoning-effort", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    store = ReplayStore(args.cache_dir)
    if args.import_routerarena_detail:
        imported, skipped = import_routerarena_detail(store, args.import_routerarena_detail, args.reasoning_effort)
        LOGGER.info("imported=%s skipped=%s cache_dir=%s", imported, skipped, store.directory)
        return
    server = ReplayHTTPServer(
        (args.host, args.port),
        store,
        args.mode,
        os.getenv("REPLAY_UPSTREAM_BASE_URL", ""),
        os.getenv("REPLAY_UPSTREAM_API_KEY", ""),
        int(os.getenv("REPLAY_UPSTREAM_TIMEOUT_SECONDS", "600")),
        os.getenv("REPLAY_UPSTREAM_PROXY", "").strip(),
    )
    LOGGER.info("listening on %s:%s mode=%s cache_dir=%s", args.host, args.port, args.mode, store.directory)
    server.serve_forever()


if __name__ == "__main__":
    main()
