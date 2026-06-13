#!/usr/bin/env bash
#SBATCH --job-name=dd-round5
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# Round 5: ODR fix + overlap. Expect cross-GPU BITWISE now, and weak-scaling
# efficiency >= 95% with --overlap.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "=== pytest gate ==="
${PYTHON_BIN} -m pytest test/test_stepped_forward.py test/test_stepped_backward.py \
    test/test_dd_two_tile.py test/test_dd_tiles_3d.py test/test_dd_backward_two_tile.py \
    -q 2>&1 | tail -3

echo "=== cross-GPU determinism (expect BITWISE after ODR fix) ==="
${TORCHRUN} --nproc-per-node=4 test/dd_cross_gpu_diag.py

echo "=== NCCL checks (expect PASS bit-level) ==="
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8 --nz 600 --nx 2240 --nt 600 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --ndim 3 --nz 48 --ny 40 --nx 112 --nt 100 --abcn 8 --fast

echo "=== WEAK SCALING 2D 4096^2/GPU: serial vs overlap ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --overlap
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --overlap

echo "=== WEAK SCALING 3D 320^3/GPU: serial vs overlap ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --overlap
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --overlap

echo "=== STRONG SCALING 2D 4096x8192 with overlap ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --nz 4096 --nxp 8192 --nt 400
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --overlap
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 2048 --nt 400 --overlap

echo "DONE"
