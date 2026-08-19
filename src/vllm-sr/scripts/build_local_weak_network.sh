#!/usr/bin/env bash
set -euo pipefail

# Build and optionally start vllm-sr in weak-network environments.
#
# What this script does:
# 1) Creates a temporary Dockerfile with mirror/proxy rewrites.
# 2) Removes heavyweight optional Python dependency installs for faster build.
# 3) Builds local image (CPU: ghcr.io/vllm-project/semantic-router/vllm-sr:latest
#    | GPU: semantic-router:cuda via tools/docker/Dockerfile.extproc-cuda).
# 4) Optionally starts the stack.
#
# GPU mode (NVIDIA, ONNX Runtime CUDA EP):
#   VLLM_SR_GPU=1 ./scripts/build_local_weak_network.sh
#   - Builds semantic-router:cuda (ONNX-only, cuda-dynamic feature).
#   - Starts with `docker run --gpus all -e AI_BINDING=onnx`.
#   - API port honors VLLM_SR_PORT_OFFSET (default 200) -> 8280 when 8080 is taken.
#   - Models are bind-mounted from src/vllm-sr/models to /app/models.
#
# Env knobs:
#   VLLM_SR_GPU          - set to 1 for NVIDIA GPU (ONNX+CUDA) build/start
#   VLLM_SR_IMAGE        - override built image name (default per mode)
#   VLLM_SR_API_PORT     - base API port for GPU mode (default 8080; +offset = published)
#   VLLM_SR_CONFIG       - config file for GPU mode (default src/vllm-sr/config.yaml)
#   VLLM_SR_STACK_NAME / VLLM_SR_PORT_OFFSET / START_AFTER_BUILD / HF_TOKEN
#   GITHUB_MIRROR_PREFIX - optional github mirror for candle builds (CPU mode only)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

GPU_MODE="${VLLM_SR_GPU:-0}"

TMP_DOCKERFILE="/tmp/Dockerfile.vllm-sr.local"
IMAGE="${VLLM_SR_IMAGE:-ghcr.io/vllm-project/semantic-router/vllm-sr:latest}"
if [[ "${GPU_MODE}" == "1" ]]; then
	IMAGE="${VLLM_SR_IMAGE:-semantic-router:cuda}"
	GPU_DOCKERFILE="${REPO_ROOT}/tools/docker/Dockerfile.extproc-cuda"
fi

STACK_NAME="${VLLM_SR_STACK_NAME:-lane-b}"
PORT_OFFSET="${VLLM_SR_PORT_OFFSET:-200}"
START_AFTER_BUILD="${START_AFTER_BUILD:-1}"
GITHUB_MIRROR_PREFIX="${GITHUB_MIRROR_PREFIX:-}"

# GPU mode: build the ONNX+CUDA image directly (Dockerfile already carries
# weak-network mirror fixes: goproxy.cn / rsproxy.cn / aliyun PyPI).
if [[ "${GPU_MODE}" == "1" ]]; then
	echo "[1/4] GPU mode: using ${GPU_DOCKERFILE}"
	cp "${GPU_DOCKERFILE}" "${TMP_DOCKERFILE}"

	echo "[2/4] Build GPU image: ${IMAGE}"
	pkill -f "docker build .*${IMAGE}.*${TMP_DOCKERFILE}" 2>/dev/null || true
	cd "${REPO_ROOT}"
	docker build --network host --progress=plain \
		--build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn \
		--build-arg GOPROXY=https://goproxy.cn,direct \
		-t "${IMAGE}" -f "${TMP_DOCKERFILE}" .

	echo "[3/4] Verify image exists"
	docker image inspect "${IMAGE}" >/dev/null
	echo "Image OK: ${IMAGE}"

	if [[ "${START_AFTER_BUILD}" != "1" ]]; then
		echo "[4/4] Skip start (START_AFTER_BUILD=${START_AFTER_BUILD})"
		exit 0
	fi

	# GPU run: host network + --gpus all + AI_BINDING=onnx.
	# Publish API on base port + offset (8080+200=8280) to avoid conflicts.
	GPU_CONFIG="${VLLM_SR_CONFIG:-${ROOT_DIR}/config.yaml}"
	GPU_API_BASE="${VLLM_SR_API_PORT:-8080}"
	GPU_API_PUBLISHED=$((GPU_API_BASE + PORT_OFFSET))
	GPU_CONTAINER="sr-${STACK_NAME}"
	MODELS_DIR="${ROOT_DIR}/models"

	echo "[4/4] Start GPU stack: ${IMAGE} (api ${GPU_API_PUBLISHED})"
	docker rm -f "${GPU_CONTAINER}" 2>/dev/null || true
	docker run -d --name "${GPU_CONTAINER}" \
		--network host \
		--gpus all \
		-e AI_BINDING=onnx \
		-v "${GPU_CONFIG}:/app/config/config.yaml:ro" \
		-v "${MODELS_DIR}:/app/models" \
		"${IMAGE}" \
		--api-port="${GPU_API_PUBLISHED}" >/dev/null

	echo "GPU container started: ${GPU_CONTAINER}"
	echo "  API:      http://localhost:${GPU_API_PUBLISHED}/api/v1/classify/intent"
	echo "  Verify:   docker logs -f ${GPU_CONTAINER} | grep 'Using CUDA execution provider'"
	exit 0
fi

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

# Harden git/cargo against flaky GitHub links (notably candle git dependency).
perl -0777 -i -pe 's#\[ "\$GIT_SSL_NO_VERIFY" = "1" \] && git config --global http\.sslVerify false \|\| true;#[ "\$GIT_SSL_NO_VERIFY" = "1" ] && git config --global http.sslVerify false || true; git config --global http.version HTTP/1.1 || true; git config --global http.postBuffer 524288000 || true; git config --global http.lowSpeedLimit 1000 || true; git config --global http.lowSpeedTime 60 || true;#g' "${TMP_DOCKERFILE}"

# Optional GitHub mirror rewrite (disabled by default; enable only when mirror is reachable).
if [[ -n "${GITHUB_MIRROR_PREFIX}" ]]; then
	sed -i "s#git config --global http.lowSpeedTime 60 || true;#git config --global http.lowSpeedTime 60 || true; git config --global url.\\\"${GITHUB_MIRROR_PREFIX}\\\".insteadof \\\"https://github.com/\\\" || true;#g" "${TMP_DOCKERFILE}"
	echo "Using optional GitHub mirror prefix: ${GITHUB_MIRROR_PREFIX}"
fi
sed -i 's#ENV CARGO_REGISTRIES_CRATES_IO_INDEX=https://rsproxy.cn/crates.io-index#ENV CARGO_REGISTRIES_CRATES_IO_INDEX=https://rsproxy.cn/crates.io-index\nENV CARGO_NET_RETRY=20\nENV CARGO_HTTP_TIMEOUT=120\nENV CARGO_HTTP_MULTIPLEXING=false#g' "${TMP_DOCKERFILE}"

# Add bounded retries around cargo build commands.
sed -i 's#cargo build --release --no-default-features --target aarch64-unknown-linux-gnu;#for i in 1 2 3 4 5; do cargo build --release --no-default-features --target aarch64-unknown-linux-gnu \&\& break; if [ \$i -eq 5 ]; then exit 1; fi; echo "cargo retry \$i\/5 (candle arm64)"; sleep \$((i*10)); done;#g' "${TMP_DOCKERFILE}"
sed -i 's#cargo build --release --no-default-features;#for i in 1 2 3 4 5; do cargo build --release --no-default-features \&\& break; if [ \$i -eq 5 ]; then exit 1; fi; echo "cargo retry \$i\/5 (candle amd64)"; sleep \$((i*10)); done;#g' "${TMP_DOCKERFILE}"
sed -i 's#cargo build --release --target aarch64-unknown-linux-gnu;#for i in 1 2 3 4 5; do cargo build --release --target aarch64-unknown-linux-gnu \&\& break; if [ \$i -eq 5 ]; then exit 1; fi; echo "cargo retry \$i\/5 (rust arm64)"; sleep \$((i*10)); done;#g' "${TMP_DOCKERFILE}"
sed -i 's#cargo build --release;#for i in 1 2 3 4 5; do cargo build --release \&\& break; if [ \$i -eq 5 ]; then exit 1; fi; echo "cargo retry \$i\/5 (rust amd64)"; sleep \$((i*10)); done;#g' "${TMP_DOCKERFILE}"
sed -i 's#OPENSSL_LIB_DIR=/usr/lib/aarch64-linux-gnu \\#OPENSSL_LIB_DIR=/usr/lib/aarch64-linux-gnu; \\#g' "${TMP_DOCKERFILE}"

# Final runtime stage apt mirror + leaner package set.
sed -i '0,/RUN set -eux; \\/s#RUN set -eux; \\#RUN set -eux; \\\n    sed -i "s|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list.d/debian.sources || true; \\#' "${TMP_DOCKERFILE}"
sed -i '/^[[:space:]]*docker\.io;[[:space:]]*\\$/d' "${TMP_DOCKERFILE}"
sed -i 's/^[[:space:]]*ca-certificates[[:space:]]*\\$/        ca-certificates; \\/' "${TMP_DOCKERFILE}"

# Remove heavyweight optional deps for weak-network local runtime builds.
sed -i '/COPY bench\/ \/app\/bench\//d' "${TMP_DOCKERFILE}"
sed -i '/COPY src\/training\/model_eval\//d' "${TMP_DOCKERFILE}"
sed -i '/\/app\/bench\/requirements.txt/d' "${TMP_DOCKERFILE}"
sed -i '/\/app\/src\/training\/model_eval\/requirements.txt/d' "${TMP_DOCKERFILE}"
sed -i 's#pip install \$PIP_EXTRA --no-cache-dir -r requirements.txt && \\#pip install \$PIP_EXTRA --no-cache-dir -r requirements.txt#' "${TMP_DOCKERFILE}"

# Harden pip install against slow/unstable links in weak-network environments.
# NOTE: pypi.tuna.tsinghua.edu.cn intermittently returns HTTP 403 for wheel
# downloads from inside the build container; use mirrors.aliyun.com instead.
sed -i 's#pip install \$PIP_EXTRA --no-cache-dir -r requirements.txt#pip install \$PIP_EXTRA --no-cache-dir --retries 50 --timeout 600 --default-timeout 600 --progress-bar off -i https:\/\/mirrors.aliyun.com\/pypi\/simple --trusted-host mirrors.aliyun.com -r requirements.txt#' "${TMP_DOCKERFILE}"

echo "[2/5] Stop duplicate builds (if any)"
pkill -f "docker build .*${IMAGE}.*${TMP_DOCKERFILE}" || true
pkill -f "docker-buildx buildx build .*${IMAGE}.*${TMP_DOCKERFILE}" || true

echo "[3/5] Build local image: ${IMAGE}"
cd "${REPO_ROOT}"
docker build --network host --progress=plain --build-arg GIT_SSL_NO_VERIFY=1 -t "${IMAGE}" -f "${TMP_DOCKERFILE}" .

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
