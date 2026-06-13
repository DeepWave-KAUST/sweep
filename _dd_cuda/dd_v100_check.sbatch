#!/usr/bin/env bash
#SBATCH --job-name=dd-v100-check
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# Runs the prebuilt (arch 7.0) extension — no compiler needed at runtime.
set -uo pipefail

WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python

cd "${WORK}"
echo "=== host $(hostname), GPUs: ==="
nvidia-smi -L

export PYTHONPATH="${WORK}/src"

echo "=== single-GPU equivalence pytest on V100 ==="
${PYTHON_BIN} -m pytest test/test_stepped_forward.py test/test_dd_two_tile.py -q 2>&1 | tail -4

echo "=== NCCL 2-rank ==="
${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=2 \
    test/dd_nccl_check.py --abcn 8

echo "=== NCCL 4-rank ==="
${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=4 \
    test/dd_nccl_check.py --abcn 8

echo "=== NCCL 4-rank, bigger grid (first scaling look) ==="
${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=4 \
    test/dd_nccl_check.py --abcn 8 --nz 600 --nx 2240 --nt 600

echo "DONE"
