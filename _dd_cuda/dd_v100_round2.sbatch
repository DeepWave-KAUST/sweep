#!/usr/bin/env bash
#SBATCH --job-name=dd-round2
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "=== full pytest gate on V100 (rebuilt .so, incl. backward) ==="
${PYTHON_BIN} -m pytest test/test_stepped_forward.py test/test_stepped_backward.py \
    test/test_dd_two_tile.py test/test_dd_tiles_3d.py test/test_dd_backward_two_tile.py \
    -q 2>&1 | tail -6

echo "=== diag: cross-GPU determinism + micro-bench (now incl. fast_halo) ==="
${TORCHRUN} --nproc-per-node=4 test/dd_cross_gpu_diag.py

echo "=== NCCL check, std vs fast ==="
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8 --nz 600 --nx 2240 --nt 600 --fast

echo "=== WEAK SCALING with fast exchanger: 2D 4096^2/GPU ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --fast

echo "=== WEAK SCALING with fast exchanger: 3D 320^3/GPU ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --fast

echo "=== STRONG SCALING with fast exchanger: 2D 4096x8192 ==="
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 2048 --nt 400 --fast

echo "DONE"
