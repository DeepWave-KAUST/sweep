#!/usr/bin/env bash
#SBATCH --job-name=dd-diag-bench
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "=== full pytest gate on V100 (incl. backward + DD grad) ==="
${PYTHON_BIN} -m pytest test/test_stepped_forward.py test/test_stepped_backward.py \
    test/test_dd_two_tile.py test/test_dd_tiles_3d.py test/test_dd_backward_two_tile.py \
    -q 2>&1 | tail -3

echo "=== cross-GPU determinism + exchange micro-bench (4 ranks) ==="
${TORCHRUN} --nproc-per-node=4 test/dd_cross_gpu_diag.py

echo "=== NCCL check 2/4 rank (graded bit/tol) ==="
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_check.py --abcn 8
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --abcn 8 --nz 600 --nx 2240 --nt 600

echo "=== M6 WEAK SCALING: 2D tile 4096x4096, nt=400 ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400

echo "=== M6 WEAK SCALING: 3D tile 320x320x320, nt=150 ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150

echo "=== M6 STRONG SCALING: 2D global 4096x8192, nt=400 ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --nz 4096 --nxp 8192 --nt 400
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_bench.py --nz 4096 --nxp 4096 --nt 400
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --nz 4096 --nxp 2048 --nt 400

echo "DONE"
