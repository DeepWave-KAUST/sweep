#!/usr/bin/env bash
#SBATCH --job-name=dd-round6
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# Final efficiency accounting: none (= slowest-card baseline, all 4 GPUs
# computing with zero comm) vs serial vs overlap IN THE SAME JOB, so the
# denominator carries the same card mix as the numerator. Repeated twice.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

for round in 1 2; do
  echo "=== ROUND ${round}: 2D 4096^2/GPU px=4 ==="
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --exchange-mode none
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --exchange-mode full
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --overlap
  echo "=== ROUND ${round}: 3D 320^3/GPU px=4 ==="
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --exchange-mode none
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --exchange-mode full
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --overlap
done

echo "DONE"
