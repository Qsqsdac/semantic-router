#!/usr/bin/env python3
"""Tiny OpenAI-compatible backend for semantic-cache route testing."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


STATE = {
    "chat_completions": 0,
    "requests": [],
}


def _extract_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(parts)
    return ""


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-openai/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write_json(200, {"ok": True})
            return
        if self.path == "/stats":
            self._write_json(200, STATE)
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0") or "0")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_json(400, {"error": "invalid json"})
            return

        if self.path == "/stats/reset":
            STATE["chat_completions"] = 0
            STATE["requests"] = []
            self._write_json(200, {"ok": True})
            return

        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._write_json(404, {"error": "not found", "path": self.path})
            return

        STATE["chat_completions"] += 1
        call_id = STATE["chat_completions"]
        model = str(payload.get("model") or "mock-model")
        user_text = _extract_user_text(payload)
        STATE["requests"].append(
            {
                "call_id": call_id,
                "model": model,
                "text_preview": user_text[:160],
                "text_length": len(user_text),
            }
        )
        if len(STATE["requests"]) > 20:
            STATE["requests"] = STATE["requests"][-20:]

        now = int(time.time())
        self._write_json(
            200,
            {
                "id": f"chatcmpl-mock-{call_id}",
                "object": "chat.completion",
                "created": now,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"mock response #{call_id}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(user_text.split())),
                    "completion_tokens": 4,
                    "total_tokens": max(1, len(user_text.split())) + 4,
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock OpenAI backend listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
