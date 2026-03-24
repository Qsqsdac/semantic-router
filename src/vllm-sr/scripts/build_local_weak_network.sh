#!/usr/bin/env bash
set -euo pipefail

# Build and optionally start vllm-sr in weak-network environments.
#
# What this script does:
# 1) Creates a temporary Dockerfile with mirror/proxy rewrites.
# 2) Removes heavyweight optional Python dependency installs for faster build.
# 3) Builds local image: ghcr.io/vllm-project/semantic-router/vllm-sr:latest
# 4) Optionally starts vllm-sr with --image-pull-policy never.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TMP_DOCKERFILE="/tmp/Dockerfile.vllm-sr.local"
IMAGE="ghcr.io/vllm-project/semantic-router/vllm-sr:latest"

STACK_NAME="${VLLM_SR_STACK_NAME:-lane-b}"
PORT_OFFSET="${VLLM_SR_PORT_OFFSET:-200}"
START_AFTER_BUILD="${START_AFTER_BUILD:-1}"

echo "[1/5] Prepare temporary Dockerfile: ${TMP_DOCKERFILE}"
cp "${ROOT_DIR}/Dockerfile" "${TMP_DOCKERFILE}"

# Use reachable base-image mirrors.
sed -i 's#FROM --platform=\$BUILDPLATFORM rustlang/rust:nightly#FROM --platform=$BUILDPLATFORM docker.m.daocloud.io/rustlang/rust:nightly#g' "${TMP_DOCKERFILE}"
sed -i 's#FROM --platform=\$BUILDPLATFORM golang:1.24#FROM --platform=$BUILDPLATFORM docker.m.daocloud.io/library/golang:1.24#g' "${TMP_DOCKERFILE}"
sed -i 's#FROM node:20-alpine#FROM docker.m.daocloud.io/library/node:20-alpine#g' "${TMP_DOCKERFILE}"
sed -i 's#FROM envoyproxy/envoy:v1.34-latest#FROM docker.m.daocloud.io/envoyproxy/envoy:v1.34-latest#g' "${TMP_DOCKERFILE}"
sed -i 's#^FROM python:3.12-slim#FROM docker.m.daocloud.io/library/python:3.12-slim#g' "${TMP_DOCKERFILE}"

# Go modules via goproxy.
sed -i 's#ENV GOPROXY=https://proxy.golang.org,direct#ENV GOPROXY=https://goproxy.cn,direct\nENV GOSUMDB=off#g' "${TMP_DOCKERFILE}"
sed -i 's#RUN go mod download#RUN GOPROXY=https://goproxy.cn,direct GOSUMDB=off go mod download#g' "${TMP_DOCKERFILE}"
sed -i 's# go build # GOPROXY=https://goproxy.cn,direct GOSUMDB=off go build #g' "${TMP_DOCKERFILE}"

# NPM mirror.
sed -i 's#https://registry.npmjs.org/#https://registry.npmmirror.com/#g' "${TMP_DOCKERFILE}"

# Cargo mirror settings.
perl -0777 -i -pe 's#ENV CARGO_NET_GIT_FETCH_WITH_CLI=true#ENV CARGO_NET_GIT_FETCH_WITH_CLI=true\nENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse\nENV CARGO_REGISTRIES_CRATES_IO_INDEX=https://rsproxy.cn/crates.io-index#g' "${TMP_DOCKERFILE}"

# Final runtime stage apt mirror + leaner package set.
sed -i '0,/RUN set -eux; \\/s#RUN set -eux; \\#RUN set -eux; \\\n    sed -i "s|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list.d/debian.sources || true; \\#' "${TMP_DOCKERFILE}"
sed -i '/^[[:space:]]*docker\.io;[[:space:]]*\\$/d' "${TMP_DOCKERFILE}"
sed -i 's/^[[:space:]]*ca-certificates[[:space:]]*\\$/        ca-certificates; \\/' "${TMP_DOCKERFILE}"

# Remove heavyweight optional deps for weak-network local runtime builds.
sed -i '/COPY bench\/ \/app\/bench\//d' "${TMP_DOCKERFILE}"
sed -i '/COPY src\/training\/model_eval\//d' "${TMP_DOCKERFILE}"
perl -0777 -i -pe 's#pip install \$PIP_EXTRA --no-cache-dir -r requirements.txt &&\s*pip install \$PIP_EXTRA --no-cache-dir -r /app/bench/requirements.txt &&\s*pip install \$PIP_EXTRA --no-cache-dir -r /app/src/training/model_eval/requirements.txt#pip install $PIP_EXTRA --no-cache-dir -r requirements.txt#g' "${TMP_DOCKERFILE}"
perl -0777 -i -pe 's#pip install \$PIP_EXTRA --no-cache-dir -r requirements.txt &&\s*pip install \$PIP_EXTRA --no-cache-dir -r /app/bench/requirements.txt#pip install $PIP_EXTRA --no-cache-dir -r requirements.txt#g' "${TMP_DOCKERFILE}"

echo "[2/5] Stop duplicate builds (if any)"
pkill -f "docker build .*${IMAGE}.*${TMP_DOCKERFILE}" || true
pkill -f "docker-buildx buildx build .*${IMAGE}.*${TMP_DOCKERFILE}" || true

echo "[3/5] Build local image: ${IMAGE}"
cd "${ROOT_DIR}/.."
docker build --network host --progress=plain -t "${IMAGE}" -f "${TMP_DOCKERFILE}" .

echo "[4/5] Verify image exists"
docker image inspect "${IMAGE}" >/dev/null
echo "Image OK: ${IMAGE}"

if [[ "${START_AFTER_BUILD}" != "1" ]]; then
	echo "[5/5] Skip start (START_AFTER_BUILD=${START_AFTER_BUILD})"
	exit 0
fi

echo "[5/5] Start local stack with never pull"
cd "${ROOT_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
	HF_TOKEN="${HF_TOKEN}" \
	VLLM_SR_STACK_NAME="${STACK_NAME}" \
	VLLM_SR_PORT_OFFSET="${PORT_OFFSET}" \
	vllm-sr serve --image-pull-policy never
else
	VLLM_SR_STACK_NAME="${STACK_NAME}" \
	VLLM_SR_PORT_OFFSET="${PORT_OFFSET}" \
	vllm-sr serve --image-pull-policy never
fi
