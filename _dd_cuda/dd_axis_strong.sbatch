#!/usr/bin/env bash
#SBATCH --job-name=dd-axisstrong
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=00:50:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-padopt/_dd_cuda/slurm-axisstrong-%j.log

# STRONG-scaling decomposition-axis comparison via the PRODUCTION DDPropagator
# (correct multi-axis/corner halo + acoustic overlap), end-to-end forward.
# Same global 256x256x1024 on 8 GPUs. The none-mode shape sweep showed the
# px4py2 balanced tile has the BEST compute floor (0.608 ms, +8.7% vs x-cut's
# 0.666): cutting more axes => more cut-aware PML savings. Question: does the
# extra comm of a 2-D cut eat that compute win, or does balanced win overall?
# (px2py4 / y-cut crash in the boundary-save kernel for thin-y tiles, ny<=64,
# so they are excluded — and were slower in compute anyway.)
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PY} -m torch.distributed.run --standalone"
cd "${WORK}"
export PYTHONPATH="${WORK}/src"
echo "=== host $(hostname) ==="; nvidia-smi -L | head -1
C=test/dd_ddp_timing.py
NT=200

echo "@@@ px1 single-GPU reference (strong denominator)"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${C} --px 1 --py 1 --nz 256 --ny 256 --nx 1024 --nt ${NT}

echo "@@@ x-cut px8 py1 (current default; 1-D x exchange + overlap)"
${TR} --nproc-per-node=8 ${C} --px 8 --py 1 --nz 256 --ny 256 --nx 1024 --nt ${NT}

echo "@@@ balanced px4 py2 (2-D cut; best compute floor)"
${TR} --nproc-per-node=8 ${C} --px 4 --py 2 --nz 256 --ny 256 --nx 1024 --nt ${NT}

echo "@@@ repeat x-cut + balanced back-to-back (A/B in one hot allocation)"
${TR} --nproc-per-node=8 ${C} --px 8 --py 1 --nz 256 --ny 256 --nx 1024 --nt ${NT}
${TR} --nproc-per-node=8 ${C} --px 4 --py 2 --nz 256 --ny 256 --nx 1024 --nt ${NT}

echo "DONE"
