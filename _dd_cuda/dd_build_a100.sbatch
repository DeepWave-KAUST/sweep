#!/usr/bin/env bash
#SBATCH --job-name=dd-build
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/build-%j.log

set -uo pipefail

WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python

# module tree on these nodes lacks cuda/*; use the install root directly
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary
export PATH="${CUDA_HOME}/bin:${PATH}"
nvcc --version | tail -2

cd "${WORK}"
# rsync preserves source mtimes which can be OLDER than cached build/*.o
# (clock skew) — ninja then silently skips recompilation. Force full build.
rm -rf build
export SWEEP_BUILD_CUDA=1
export SWEEP_CUDA_ARCH_LIST="7.0"
export TORCH_CUDA_ARCH_LIST="7.0"
${PYTHON_BIN} setup.py build_ext --inplace 2>&1 | tail -3
ls -la src/sweep/_C*.so && echo "BUILD_OK"
