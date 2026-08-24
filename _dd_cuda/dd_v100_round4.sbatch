#!/usr/bin/env bash
#SBATCH --job-name=dd-round4
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# M6 decomposition experiment:
#  (1) per-GPU baselines -> true slowest-card denominator for weak scaling
#  (2) px=4 with exchange-mode none/stage/full -> where do the extra us/step go
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "=== per-GPU single baselines, 2D 4096^2 ==="
for g in 0 1 2 3; do
  echo "--- GPU ${g} ---"
  CUDA_VISIBLE_DEVICES=${g} ${TORCHRUN} --nproc-per-node=1 \
      test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400
done

echo "=== per-GPU single baselines, 3D 320^3 ==="
for g in 0 1 2 3; do
  echo "--- GPU ${g} ---"
  CUDA_VISIBLE_DEVICES=${g} ${TORCHRUN} --nproc-per-node=1 \
      test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150
done

echo "=== px=4 exchange decomposition, 2D ==="
for m in none stage full; do
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py \
      --nz 4096 --nxp 4096 --nt 400 --exchange-mode ${m}
done

echo "=== px=4 exchange decomposition, 3D ==="
for m in none stage full; do
  ${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py \
      --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --exchange-mode ${m}
done

echo "DONE"
