#!/usr/bin/env bash
# Run the local Qwen3.5-4B service only while openai-replay is active.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPLAY_PYTHON=${REPLAY_PYTHON:?Set REPLAY_PYTHON to the interpreter for openai-replay.}
QWEN35_4B_PYTHON=${QWEN35_4B_PYTHON:?Set QWEN35_4B_PYTHON to the interpreter for the local Qwen server.}
QWEN35_4B_SERVICE_DIR=${QWEN35_4B_SERVICE_DIR:?Set QWEN35_4B_SERVICE_DIR to the qwen35-4b-service directory.}
QWEN35_4B_HOST=${QWEN35_4B_HOST:-127.0.0.1}
QWEN35_4B_PORT=${QWEN35_4B_PORT:-18080}
QWEN35_4B_STARTUP_TIMEOUT_SECONDS=${QWEN35_4B_STARTUP_TIMEOUT_SECONDS:-180}

qwen_pid=""
replay_pid=""

is_running() {
    [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null
}

stop_process() {
    local pid=$1
    if is_running "$pid"; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    trap - EXIT INT TERM HUP TSTP CONT
    stop_process "$replay_pid"
    stop_process "$qwen_pid"
}

suspend_services() {
    if is_running "$replay_pid"; then
        kill -STOP "$replay_pid"
    fi
    if is_running "$qwen_pid"; then
        kill -STOP "$qwen_pid"
    fi
    kill -STOP "$$"
}

resume_services() {
    if is_running "$qwen_pid"; then
        kill -CONT "$qwen_pid"
    fi
    if is_running "$replay_pid"; then
        kill -CONT "$replay_pid"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
trap suspend_services TSTP
trap resume_services CONT

"$QWEN35_4B_PYTHON" "$QWEN35_4B_SERVICE_DIR/server.py" \
    --host "$QWEN35_4B_HOST" \
    --port "$QWEN35_4B_PORT" &
qwen_pid=$!

deadline=$((SECONDS + QWEN35_4B_STARTUP_TIMEOUT_SECONDS))
until curl -fsS --max-time 2 "http://${QWEN35_4B_HOST}:${QWEN35_4B_PORT}/health" >/dev/null; do
    if ! is_running "$qwen_pid"; then
        wait "$qwen_pid"
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for local Qwen3.5-4B at ${QWEN35_4B_HOST}:${QWEN35_4B_PORT}" >&2
        exit 1
    fi
    sleep 1
done

"$REPLAY_PYTHON" "$SCRIPT_DIR/openai_record_replay_backend.py" "$@" &
replay_pid=$!
wait "$replay_pid"