#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"
MODEL_DIR="${PROJECT_DIR}/model"
GRASPNETAPI_VERSION="${GRASPNETAPI_VERSION:-1.2.11}"

FASTSAM_MODEL_NAME="FastSAM-x.pt"
YOLOWORLD_MODEL_NAME="yolov8x-worldv2.pt"

FASTSAM_URL="${FASTSAM_URL:-https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-x.pt}"
YOLOWORLD_URL="${YOLOWORLD_URL:-https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt}"

log() {
    printf '[setup] %s\n' "$*"
}

die() {
    printf '[setup][error] %s\n' "$*" >&2
    exit 1
}

ensure_python() {
    command -v python >/dev/null 2>&1 || die "python was not found in the current environment."
    python -m pip --version >/dev/null 2>&1 || die "pip was not found in the current Python environment."
}

install_requirements() {
    [[ -f "${REQUIREMENTS_FILE}" ]] || die "requirements.txt not found: ${REQUIREMENTS_FILE}"

    log "using Python: $(python -c 'import sys; print(sys.executable)')"
    log "installing pip tooling."
    python -m pip install --upgrade pip setuptools wheel

    log "installing Python dependencies from requirements.txt."
    python -m pip install -r "${REQUIREMENTS_FILE}"

    # graspnetAPI pins numpy==1.20.3 in its package metadata, which conflicts
    # with modern MuJoCo/Open3D/Python environments. Install it without deps
    # after the main environment has selected compatible numeric packages.
    log "installing graspnetAPI ${GRASPNETAPI_VERSION} without dependencies."
    python -m pip install --no-deps "graspnetAPI==${GRASPNETAPI_VERSION}"
}

download_file() {
    local url="$1"
    local target="$2"
    local tmp="${target}.part"

    rm -f "${tmp}"

    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 3 --connect-timeout 30 -o "${tmp}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 -O "${tmp}" "${url}"
    else
        die "neither curl nor wget is available; cannot download ${target}."
    fi

    [[ -s "${tmp}" ]] || die "download produced an empty file: ${target}"
    mv "${tmp}" "${target}"
}

download_model_if_missing() {
    local name="$1"
    local url="$2"
    local target="${MODEL_DIR}/${name}"

    mkdir -p "${MODEL_DIR}"

    if [[ -s "${target}" ]]; then
        log "model already exists; skipping: ${target}"
        return
    fi

    if [[ -e "${target}" ]]; then
        log "removing empty or incomplete model file: ${target}"
        rm -f "${target}"
    fi

    log "downloading ${name} to ${MODEL_DIR}."
    download_file "${url}" "${target}"
    log "downloaded ${name}."
}

main() {
    cd "${PROJECT_DIR}"
    trap 'rm -f "${MODEL_DIR}"/*.part 2>/dev/null || true' EXIT

    ensure_python
    install_requirements

    download_model_if_missing "${FASTSAM_MODEL_NAME}" "${FASTSAM_URL}"
    download_model_if_missing "${YOLOWORLD_MODEL_NAME}" "${YOLOWORLD_URL}"

    log "setup complete."
}

main "$@"
