#!/usr/bin/env bash
#SBATCH --job-name=dd-round7
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# Spatial-order sweep: weak-scaling efficiency for so in {2,4,6,8}
# (halo M = so/2), none/full/overlap each, 2D + 3D, px=4.
# Plus order-8 NCCL correctness checks.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "=== order-8 NCCL correctness (2D + 3D, 4 ranks) ==="
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8 --so 8 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --ndim 3 --nz 48 --ny 40 --nx 112 --nt 100 --abcn 8 --so 8 --fast

for so in 2 4 6 8; do
  echo "=== so=${so}: 2D 4096^2/GPU px=4 ==="
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --so ${so} --exchange-mode none
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --so ${so} --exchange-mode full
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --so ${so} --overlap
  echo "=== so=${so}: 3D 320^3/GPU px=4 ==="
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --so ${so} --exchange-mode none
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --so ${so} --exchange-mode full
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --so ${so} --overlap
done

echo "DONE"
