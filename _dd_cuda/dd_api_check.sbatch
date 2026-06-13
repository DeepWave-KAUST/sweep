#!/usr/bin/env bash
#SBATCH --job-name=dd-api-check
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-api-%j.log

# Multi-GPU validation of the DDPropagator one-call API vs single-domain
# reference (forward record + per-tile gradient), bitwise. Pure-Python add —
# reuses the multiarch .so already built for sweep-dd-cuda.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PY=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PY} -m torch.distributed.run --standalone"
cd "${WORK}"
export PYTHONPATH="${WORK}/src"
echo "=== host ==="; hostname; nvidia-smi -L | head -1

C=test/dd_api_check.py
for p in 2 4 8; do
  for fam in acoustic elastic; do
    for nd in 2 3; do
      echo "@@@ px=$p $fam ${nd}D"
      ${TR} --nproc-per-node=${p} ${C} --family ${fam} --ndim ${nd}
    done
  done
done

echo "@@@ free surface (px=4)"
${TR} --nproc-per-node=4 ${C} --family acoustic --ndim 2 --free-surface
${TR} --nproc-per-node=4 ${C} --family elastic  --ndim 2 --free-surface
${TR} --nproc-per-node=4 ${C} --family elastic  --ndim 3 --free-surface
echo DONE
