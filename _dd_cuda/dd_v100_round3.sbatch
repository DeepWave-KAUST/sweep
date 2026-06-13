#!/usr/bin/env bash
#SBATCH --job-name=dd-round3
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# M5 multi-GPU: 3-D NCCL correctness + out-of-memory demo.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TORCHRUN="${PYTHON_BIN} -m torch.distributed.run --standalone"

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "=== 3D NCCL check (x-split), 2 and 4 ranks, fast exchanger ==="
${TORCHRUN} --nproc-per-node=2 test/dd_nccl_check.py --ndim 3 --nz 48 --ny 40 --nx 56 --nt 100 --abcn 8 --fast
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_check.py --ndim 3 --nz 48 --ny 40 --nx 112 --nt 100 --abcn 8 --fast

echo "=== OOM demo part 1: single V100 CANNOT hold 768x768x1536 (expect CUDA OOM) ==="
${TORCHRUN} --nproc-per-node=1 test/dd_nccl_bench.py --ndim 3 --nz 768 --ny 768 --nxp 1536 --nt 20 --repeats 1 2>&1 | tail -3 || echo "OOM_AS_EXPECTED"

echo "=== OOM demo part 2: 4 V100s run it via DD (tile 768x768x384 each) ==="
${TORCHRUN} --nproc-per-node=4 test/dd_nccl_bench.py --ndim 3 --nz 768 --ny 768 --nxp 384 --nt 100 --repeats 2 --fast

echo "DONE"
