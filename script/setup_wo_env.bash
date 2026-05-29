#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-mujoco}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/setup.bash"

log() {
    printf '[setup-env] %s\n' "$*"
}

die() {
    printf '[setup-env][error] %s\n' "$*" >&2
    exit 1
}

ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        die "conda was not found. Please install Miniconda/Anaconda first, then rerun this script."
    fi

    # Make `conda activate` available in non-interactive bash shells.
    if ! eval "$(conda shell.bash hook 2>/dev/null)"; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null)" || die "failed to locate conda base path."
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
    fi
}

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

conda_package_installed() {
    local package="$1"
    conda list -n "${ENV_NAME}" "${package}" 2>/dev/null | awk 'NR > 3 {print $1}' | grep -Fxq "${package}"
}

create_or_activate_env() {
    ensure_conda

    if conda_env_exists; then
        log "conda environment '${ENV_NAME}' already exists; reusing it."
    else
        log "creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}."
        conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
    fi

    if conda_package_installed portaudio; then
        log "conda package 'portaudio' already installed."
    else
        log "installing PortAudio via conda-forge for PyAudio support."
        conda install -y -n "${ENV_NAME}" -c conda-forge portaudio
    fi

    conda activate "${ENV_NAME}"
    log "activated conda environment '${ENV_NAME}'."
}

main() {
    [[ -x "${SETUP_SCRIPT}" ]] || die "setup.bash is missing or not executable: ${SETUP_SCRIPT}"

    create_or_activate_env
    "${SETUP_SCRIPT}" "$@"

    log "environment setup complete."
    log "next time, run: conda activate ${ENV_NAME}"
}

main "$@"
