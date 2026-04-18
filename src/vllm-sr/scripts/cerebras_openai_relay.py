#!/usr/bin/env python3
"""OpenAI-compatible relay for Cerebras chat completions.

This service exposes a minimal subset of OpenAI API endpoints and forwards
requests to Cerebras API. It is intended as a local host-side relay so the
containerized router can call a local HTTP endpoint while the relay itself
handles outbound proxy/network policy.
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests


LOGGER = logging.getLogger("cerebras-openai-relay")

LISTEN_HOST = os.getenv("RELAY_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("RELAY_PORT", "18080"))
UPSTREAM_BASE = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/")
UPSTREAM_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
UPSTREAM_TIMEOUT = int(os.getenv("RELAY_TIMEOUT_SECONDS", "600"))
UPSTREAM_PROXY = os.getenv("RELAY_UPSTREAM_PROXY", "http://127.0.0.1:7890").strip()

MODEL_ALIASES = {
    "MoM": "llama3.1-8b",
    "MoM-large": "qwen-3-235b-a22b-instruct-2507",
}


def _proxies() -> dict[str, str] | None:
    if not UPSTREAM_PROXY:
        return None
    return {"http": UPSTREAM_PROXY, "https": UPSTREAM_PROXY}


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class RelayHandler(BaseHTTPRequestHandler):
    server_version = "CerebrasOpenAIRelay/0.1"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _forward(self, method: str, upstream_path: str, payload: dict[str, Any] | None = None) -> None:
        if not UPSTREAM_API_KEY:
            _json_response(
                self,
                500,
                {"error": {"message": "CEREBRAS_API_KEY is not set on relay", "type": "relay_config_error"}},
            )
            return

        url = f"{UPSTREAM_BASE}{upstream_path}"
        headers = {
            "Authorization": f"Bearer {UPSTREAM_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=payload,
                timeout=UPSTREAM_TIMEOUT,
                proxies=_proxies(),
            )
        except Exception as exc:
            LOGGER.exception("Upstream request failed: %s", exc)
            _json_response(
                self,
                502,
                {"error": {"message": str(exc), "type": "relay_upstream_error"}},
            )
            return

        self.send_response(resp.status_code)
        self.send_header(
            "Content-Type", resp.headers.get("Content-Type", "application/json; charset=utf-8")
        )
        self.send_header("Content-Length", str(len(resp.content)))
        self.end_headers()
        self.wfile.write(resp.content)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            _json_response(self, 200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._forward("GET", "/models")
            return
        _json_response(self, 404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            _json_response(self, 404, {"error": {"message": "not found"}})
            return

        try:
            payload = self._read_json()
        except Exception as exc:
            _json_response(self, 400, {"error": {"message": f"invalid json: {exc}"}})
            return

        requested_model = str(payload.get("model", "")).strip()
        if requested_model in MODEL_ALIASES:
            payload["model"] = MODEL_ALIASES[requested_model]

        self._forward("POST", "/chat/completions", payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), RelayHandler)
    LOGGER.info(
        "Relay listening on %s:%s, upstream=%s, proxy=%s",
        LISTEN_HOST,
        LISTEN_PORT,
        UPSTREAM_BASE,
        UPSTREAM_PROXY or "<none>",
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
