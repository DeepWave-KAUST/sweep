#!/usr/bin/env bash
#SBATCH --job-name=dd-mem-check
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-mem-%j.log

# Per-GPU memory balance of a real DD run (forward + gradient, bs). Each rank
# holds only its tile; reports per-rank peak GPU mem + imbalance.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PY=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PY} -m torch.distributed.run --standalone"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
echo "=== host ==="; hostname; nvidia-smi -L | head -1
C=test/dd_mem_check.py

${TR} --nproc-per-node=8 ${C} --family acoustic --ndim 3 --px 8 --py 1 --nz 256 --nyp 256 --nxp 256 --nt 400
${TR} --nproc-per-node=4 ${C} --family acoustic --ndim 3 --px 2 --py 2 --nz 256 --nyp 256 --nxp 256 --nt 400
${TR} --nproc-per-node=8 ${C} --family acoustic --ndim 3 --px 4 --py 2 --nz 256 --nyp 256 --nxp 256 --nt 400
${TR} --nproc-per-node=8 ${C} --family elastic  --ndim 3 --px 8 --py 1 --nz 160 --nyp 160 --nxp 160 --nt 200
${TR} --nproc-per-node=4 ${C} --family elastic  --ndim 3 --px 2 --py 2 --nz 160 --nyp 160 --nxp 160 --nt 200
${TR} --nproc-per-node=8 ${C} --family acoustic --ndim 2 --px 8 --py 1 --nz 4096 --nxp 4096 --nt 600
echo DONE
