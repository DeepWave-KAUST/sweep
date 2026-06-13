#!/usr/bin/env bash
#SBATCH --job-name=dd-round8
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# End-to-end multi-GPU NCCL DD BACKWARD (gradients) vs single domain.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "=== NCCL DD backward, 2 ranks ==="
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_backward_check.py

echo "=== NCCL DD backward, 4 ranks ==="
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_backward_check.py

echo "DONE"
