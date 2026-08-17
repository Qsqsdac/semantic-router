#!/bin/bash
set -e

CONFIG_FILE="${1:-/app/config.yaml}"
echo "Starting router from canonical config..."
echo "  Config file: $CONFIG_FILE"

echo "Starting router..."
exec /usr/local/bin/router \
    -config="$CONFIG_FILE" \
    -port=50051 \
    -enable-api=true \
    -api-port=8080
