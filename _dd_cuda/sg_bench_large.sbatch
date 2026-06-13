#!/usr/bin/env bash
#SBATCH --job-name=sg-bench-lg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/bench-%x-%j.log

# Single-GPU NON-DD regression A/B at LARGER sizes (bs mode only — full mode
# would store all timesteps and OOM at these grids). current vs origin/dev,
# back-to-back on the hot GPU. Override GPU: sbatch --gres=gpu:v100:1 / a100:1
set -uo pipefail
PY=/ibex/user/wangs0j/mambaforge/bin/python
CUR=/ibex/user/wangs0j/sweep-dd-cuda
DEV=/ibex/user/wangs0j/sweep-dev-baseline
B="${CUR}/_dd_cuda/bench_single_gpu.py"

echo "=== host ==="; hostname; nvidia-smi -L
ab() {
  echo "@@@ $*"
  PYTHONPATH="${CUR}/src" ${PY} "${B}" $* || echo "CUR run failed: $*"
  PYTHONPATH="${DEV}/src" ${PY} "${B}" $* || echo "DEV run failed: $*"
}

ab --eq acoustic2d --nz 8192 --nx 8192 --nt 500
ab --eq acoustic3d --nz 384  --ny 384  --nx 384 --nt 150
ab --eq elastic2d  --nz 4096 --nx 4096 --nt 500
ab --eq elastic3d  --nz 256  --ny 256  --nx 256 --nt 120
ab --eq acoustic2d --nz 8192 --nx 8192 --nt 500 --so 8
ab --eq elastic2d  --nz 4096 --nx 4096 --nt 500 --so 8
echo DONE
