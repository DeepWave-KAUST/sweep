#!/usr/bin/env bash
#SBATCH --job-name=sg-bench
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/bench-%x-%j.log

# Single-GPU NON-DD regression A/B: current branch (DD, gated off) vs
# origin/dev (no DD). Same hot GPU, current then dev back-to-back per config.
# Override the GPU type at submit: sbatch --gres=gpu:v100:1 / gpu:a100:1
set -uo pipefail
PY=/ibex/user/wangs0j/mambaforge/bin/python
CUR=/ibex/user/wangs0j/sweep-dd-cuda
DEV=/ibex/user/wangs0j/sweep-dev-baseline
B="${CUR}/_dd_cuda/bench_single_gpu.py"

echo "=== host ==="; hostname; nvidia-smi -L
echo "=== build tags present? ==="
ls -la "${CUR}/src/sweep/_C"*.so "${DEV}/src/sweep/_C"*.so

ab() {  # $* = bench args; run current then dev, back-to-back on the hot GPU
  echo "@@@ $*"
  PYTHONPATH="${CUR}/src" ${PY} "${B}" $* || echo "CUR run failed: $*"
  PYTHONPATH="${DEV}/src" ${PY} "${B}" $* || echo "DEV run failed: $*"
}

ab --eq acoustic2d --nz 2048 --nx 2048 --nt 500
ab --eq acoustic3d --nz 256 --ny 256 --nx 256 --nt 150
ab --eq elastic2d  --nz 1536 --nx 1536 --nt 500
ab --eq elastic3d  --nz 160 --ny 160 --nx 160 --nt 120
ab --eq acoustic2d --nz 2048 --nx 2048 --nt 500 --so 8
ab --eq elastic2d  --nz 1536 --nx 1536 --nt 500 --so 8
echo DONE
