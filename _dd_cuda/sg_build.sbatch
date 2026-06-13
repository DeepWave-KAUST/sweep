#!/usr/bin/env bash
#SBATCH --job-name=sg-build
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/build-%x-%j.log

# Multiarch (V100 sm_70 + A100 sm_80) build of one worktree, for the
# single-GPU non-DD regression A/B. Pass the worktree via --export=ALL,WORK=...
set -uo pipefail
: "${WORK:?set WORK via --export=ALL,WORK=/ibex/user/wangs0j/<worktree>}"
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary
export PATH="${CUDA_HOME}/bin:${PATH}"
nvcc --version | tail -2
cd "${WORK}"
rm -rf build                       # rsync mtime skew can make ninja skip recompile
export SWEEP_BUILD_CUDA=1
export SWEEP_CUDA_ARCH_LIST="7.0;8.0"
export TORCH_CUDA_ARCH_LIST="7.0;8.0"
${PYTHON_BIN} setup.py build_ext --inplace 2>&1 | tail -3
ls -la "${WORK}"/src/sweep/_C*.so && echo "BUILD_OK ${WORK}"
