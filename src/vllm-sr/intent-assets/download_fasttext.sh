#!/usr/bin/env bash
set -euo pipefail

# Download and build fastText CLI with weak-network friendly fallbacks.
#
# Output:
#   - ${INTENT_ASSETS_DIR}/bin/fasttext.real
#
# Existing setup compatibility:
#   - If bin/fasttext is a shim script that delegates to fasttext.real,
#     no further changes are needed.
#
# Usage:
#   ./download_fasttext.sh
#   FASTTEXT_VERSION=v0.9.2 ./download_fasttext.sh
#   FASTTEXT_SOURCE_URL=https://.../fastText-v0.9.2.tar.gz ./download_fasttext.sh
#   BUILD_JOBS=8 ./download_fasttext.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTENT_ASSETS_DIR="${INTENT_ASSETS_DIR:-${SCRIPT_DIR}}"
BIN_DIR="${INTENT_ASSETS_DIR}/bin"
CACHE_DIR="${INTENT_ASSETS_DIR}/.cache"
BUILD_DIR="${INTENT_ASSETS_DIR}/.build"
FASTTEXT_VERSION="${FASTTEXT_VERSION:-v0.9.2}"
BUILD_JOBS="${BUILD_JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 2)}"

mkdir -p "${BIN_DIR}" "${CACHE_DIR}" "${BUILD_DIR}"

log() {
  echo "[fasttext-installer] $*"
}

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "missing required command: ${cmd}"
    return 1
  fi
}

try_fetch() {
  local url="$1"
  local out="$2"
  log "trying download: ${url}"

  if command -v curl >/dev/null 2>&1; then
    if curl -fL --retry 4 --retry-delay 2 --connect-timeout 10 --max-time 1800 \
      -o "${out}.part" "${url}"; then
      mv "${out}.part" "${out}"
      return 0
    fi
  fi

  if command -v wget >/dev/null 2>&1; then
    if wget --tries=4 --timeout=30 -O "${out}.part" "${url}"; then
      mv "${out}.part" "${out}"
      return 0
    fi
  fi

  rm -f "${out}.part"
  return 1
}

install_build_deps_hint() {
  log "build dependencies not complete. Need: make + g++ + tar + (curl or wget)"
  log "example (Ubuntu): sudo apt-get update && sudo apt-get install -y build-essential curl"
}

main() {
  need_cmd tar || { install_build_deps_hint; exit 1; }
  need_cmd make || { install_build_deps_hint; exit 1; }

  if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
    install_build_deps_hint
    exit 1
  fi

  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    log "need curl or wget for download"
    exit 1
  fi

  local archive="${CACHE_DIR}/fastText-${FASTTEXT_VERSION}.tar.gz"
  local src_root=""
  local install_path="${BIN_DIR}/fasttext.real"

  if [[ ! -s "${archive}" ]]; then
    local urls=()

    if [[ -n "${FASTTEXT_SOURCE_URL:-}" ]]; then
      urls+=("${FASTTEXT_SOURCE_URL}")
    fi

    urls+=(
      "https://github.com/facebookresearch/fastText/archive/refs/tags/${FASTTEXT_VERSION}.tar.gz"
      "https://ghproxy.com/https://github.com/facebookresearch/fastText/archive/refs/tags/${FASTTEXT_VERSION}.tar.gz"
      "https://download.nju.edu.cn/github-release/facebookresearch/fastText/${FASTTEXT_VERSION}.tar.gz"
    )

    local ok=0
    for u in "${urls[@]}"; do
      if try_fetch "$u" "$archive"; then
        ok=1
        break
      fi
    done

    if [[ "$ok" -ne 1 ]]; then
      log "all download mirrors failed"
      log "you can set FASTTEXT_SOURCE_URL to an internal mirror and retry"
      exit 1
    fi
  else
    log "using cached archive: ${archive}"
  fi

  # Clean stale extracted trees from previous runs.
  rm -rf "${BUILD_DIR}/fastText-${FASTTEXT_VERSION}" "${BUILD_DIR}/fastText-${FASTTEXT_VERSION#v}"
  tar -xzf "${archive}" -C "${BUILD_DIR}"

  # Detect extracted source root robustly (v0.9.2 vs 0.9.2 naming, mirror differences).
  while IFS= read -r extracted; do
    [[ -z "${extracted}" ]] && continue
    if [[ -f "${BUILD_DIR}/${extracted}/Makefile" ]]; then
      src_root="${BUILD_DIR}/${extracted}"
      break
    fi
  done < <(tar -tzf "${archive}" | cut -d/ -f1 | sort -u)

  if [[ -z "${src_root}" ]]; then
    log "failed to detect extracted source dir with Makefile"
    log "tip: cached archive might be broken, remove ${archive} and rerun"
    exit 1
  fi

  log "building fastText from source: ${src_root}"
  make -C "${src_root}" -j"${BUILD_JOBS}"

  if [[ ! -x "${src_root}/fasttext" ]]; then
    log "build finished but binary not found: ${src_root}/fasttext"
    exit 1
  fi

  cp "${src_root}/fasttext" "${install_path}"
  chmod +x "${install_path}"
  log "installed: ${install_path}"

  # If no wrapper exists, expose the real binary as default command.
  if [[ ! -e "${BIN_DIR}/fasttext" ]]; then
    cp "${install_path}" "${BIN_DIR}/fasttext"
    chmod +x "${BIN_DIR}/fasttext"
    log "created default executable: ${BIN_DIR}/fasttext"
  else
    log "kept existing ${BIN_DIR}/fasttext (likely shim wrapper)"
  fi

  log "verification"
  local verify_out
  verify_out="$("${install_path}" 2>&1 || true)"
  if [[ "${verify_out}" != *"usage: fasttext"* ]]; then
    log "warning: installed binary failed usage check"
    exit 1
  fi

  log "done"
  log "next: put a real model at ${INTENT_ASSETS_DIR}/models/intent_fasttext.bin"
}

main "$@"
