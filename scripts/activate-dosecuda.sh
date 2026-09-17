#!/usr/bin/env bash

# Source this file from the repository root:
#   source scripts/activate-dosecuda.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced: source scripts/activate-dosecuda.sh" >&2
    exit 1
fi

dosecuda_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
spack_root="${SPACK_ROOT:-${HOME}/.spack-src}"

if [[ ! -f "${spack_root}/share/spack/setup-env.sh" ]]; then
    echo "Spack setup script not found under ${spack_root}" >&2
    return 1
fi

if [[ ! -f "${dosecuda_root}/.venv/bin/activate" ]]; then
    echo "Project virtual environment is missing: ${dosecuda_root}/.venv" >&2
    return 1
fi

source "${spack_root}/share/spack/setup-env.sh"
spack env activate "${dosecuda_root}"
source "${dosecuda_root}/.venv/bin/activate"

export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=61"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1

unset dosecuda_root spack_root
