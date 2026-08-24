#!/usr/bin/env bash
#SBATCH --job-name=dd-api-2axis
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-api-2axis-%j.log

# DDPropagator 2-axis decomposition (3-D): y-cut and 2x2, acoustic + elastic,
# vs single-domain reference (record + per-tile gradient), bitwise. Reuses the
# multiarch .so (pure-Python API).
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PY=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PY} -m torch.distributed.run --standalone"
cd "${WORK}"
export PYTHONPATH="${WORK}/src"
echo "=== host ==="; hostname; nvidia-smi -L | head -1
C=test/dd_api_check.py

echo "@@@ y-cut (px=1, py=2) world=2"
${TR} --nproc-per-node=2 ${C} --family acoustic --ndim 3 --px 1 --py 2
${TR} --nproc-per-node=2 ${C} --family elastic  --ndim 3 --px 1 --py 2

echo "@@@ y-cut (px=1, py=4) world=4"
${TR} --nproc-per-node=4 ${C} --family acoustic --ndim 3 --px 1 --py 4
${TR} --nproc-per-node=4 ${C} --family elastic  --ndim 3 --px 1 --py 4

echo "@@@ 2x2 (px=2, py=2) world=4"
${TR} --nproc-per-node=4 ${C} --family acoustic --ndim 3 --px 2 --py 2
${TR} --nproc-per-node=4 ${C} --family elastic  --ndim 3 --px 2 --py 2

echo "@@@ 2x4 (px=2, py=4) world=8"
${TR} --nproc-per-node=8 ${C} --family acoustic --ndim 3 --px 2 --py 4
${TR} --nproc-per-node=8 ${C} --family elastic  --ndim 3 --px 2 --py 4

echo "@@@ 4x2 (px=4, py=2) world=8"
${TR} --nproc-per-node=8 ${C} --family acoustic --ndim 3 --px 4 --py 2
${TR} --nproc-per-node=8 ${C} --family elastic  --ndim 3 --px 4 --py 2

echo "@@@ free surface 2x2 (px=2, py=2) world=4"
${TR} --nproc-per-node=4 ${C} --family acoustic --ndim 3 --px 2 --py 2 --free-surface
${TR} --nproc-per-node=4 ${C} --family elastic  --ndim 3 --px 2 --py 2 --free-surface
echo DONE
