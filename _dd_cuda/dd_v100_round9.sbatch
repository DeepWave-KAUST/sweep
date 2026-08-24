#!/usr/bin/env bash
#SBATCH --job-name=dd-round9
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# STRONG scaling: same global model on 1 vs 2 vs 4 GPUs.
# px=1 baselines run on GPU0 AND GPU1 (the node has two card bins, ~4%
# apart) so the speedup denominator is bracketed.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "############ 2D nz=4096, GLOBAL NX=8192 ############"
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=${g} ${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --nz 4096 --nxp 8192 --nt 400
done
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --exchange-mode full
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --overlap
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 2048 --nt 400 --exchange-mode full
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 2048 --nt 400 --overlap

echo "############ 2D nz=4096, GLOBAL NX=16384 ############"
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=${g} ${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --nz 4096 --nxp 16384 --nt 400
done
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 8192 --nt 400 --overlap
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --overlap

echo "############ 3D 320x320, GLOBAL NX=640 ############"
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=${g} ${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 640 --nt 150
done
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --exchange-mode full
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --overlap
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 160 --nt 150 --exchange-mode full
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 160 --nt 150 --overlap

echo "############ 3D 320x320, GLOBAL NX=1280 ############"
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=${g} ${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 1280 --nt 150
done
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 640 --nt 150 --overlap
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --overlap

echo "DONE"
