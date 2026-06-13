#!/usr/bin/env bash
#SBATCH --job-name=dd-mem-strong
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-mem-strong-%j.log

# Strong-scaling memory: SAME global model split across 1/2/4/8 cards
# (nxp = global_nx / px). px=1 = whole problem on one card; px=8 = each card
# holds 1/8. Shows per-card peak GPU mem dropping ~linearly with #cards.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PY=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PY} -m torch.distributed.run --standalone"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
echo "=== host ==="; hostname; nvidia-smi -L | head -1
C=test/dd_mem_check.py

echo "@@@@@ 2D acoustic STRONG: global 8192 x 8192, nt 600"
for p in 1 2 4 8; do
  ${TR} --nproc-per-node=${p} ${C} --ndim 2 --family acoustic --px ${p} --py 1 \
        --nz 8192 --nxp $((8192 / p)) --nt 600
done

echo "@@@@@ 3D acoustic STRONG: global 256 x 256 x 1024, nt 100"
for p in 1 2 4 8; do
  ${TR} --nproc-per-node=${p} ${C} --ndim 3 --family acoustic --px ${p} --py 1 \
        --nz 256 --nyp 256 --nxp $((1024 / p)) --nt 100
done

echo "@@@@@ 3D elastic STRONG: global 192 x 192 x 512, nt 120"
for p in 1 2 4 8; do
  ${TR} --nproc-per-node=${p} ${C} --ndim 3 --family elastic --px ${p} --py 1 \
        --nz 192 --nyp 192 --nxp $((512 / p)) --nt 120
done
echo DONE
