#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE_PATH=${CONFIG_FILE:-/app/config/config.yaml}
AI_BINDING=${AI_BINDING:-candle}

if [[ ! -f "$CONFIG_FILE_PATH" ]]; then
  echo "[entrypoint] Config file not found at $CONFIG_FILE_PATH" >&2
  exit 1
fi

case "$AI_BINDING" in
  onnx)
    BINARY=/app/router-onnx
    ;;
  candle|"")
    BINARY=/app/router-candle
    ;;
  *)
    echo "[entrypoint] Unknown AI_BINDING='$AI_BINDING'. Valid values: candle (default), onnx" >&2
    exit 1
    ;;
esac

if [[ ! -f "$BINARY" ]]; then
  echo "[entrypoint] Binary not found: $BINARY (AI_BINDING=$AI_BINDING)" >&2
  echo "[entrypoint] Falling back to candle binding..." >&2
  BINARY=/app/router-candle
  AI_BINDING=candle
  if [[ ! -f "$BINARY" ]]; then
    echo "[entrypoint] Fallback binary also not found: $BINARY" >&2
    exit 1
  fi
fi

echo "[entrypoint] Starting semantic-router with AI_BINDING=$AI_BINDING"
echo "[entrypoint] Config: $CONFIG_FILE_PATH"
echo "[entrypoint] Additional args: $*"

mkdir -p /etc/envoy
python3 -m cli.config_generator "$CONFIG_FILE_PATH" /etc/envoy/envoy.yaml

"$BINARY" --config "$CONFIG_FILE_PATH" "$@" &
ROUTER_PID=$!

/usr/local/bin/envoy -c /etc/envoy/envoy.yaml --log-level info &
ENVOY_PID=$!

terminate() {
  kill -TERM "$ROUTER_PID" "$ENVOY_PID" 2>/dev/null || true
  wait "$ROUTER_PID" "$ENVOY_PID" 2>/dev/null || true
}

trap terminate INT TERM

wait -n "$ROUTER_PID" "$ENVOY_PID"
STATUS=$?
terminate
exit "$STATUS"
