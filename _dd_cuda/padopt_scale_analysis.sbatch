#!/usr/bin/env bash
#SBATCH --job-name=padopt-scale
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/padopt-scale-%j.log

# Find the weak-scaling gap to 8x: compute floor (none) vs serial comm (full),
# NCCL env tuning, and larger tiles. Plus correctness re-check (serial path).
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary
export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
TR="${PY} -m torch.distributed.run --standalone"
B=test/dd_nccl_bench.py
echo "=== host $(hostname) ==="; nvidia-smi -L | head -1

echo; echo "############ BUILD multiarch ############"
rm -rf build
export SWEEP_BUILD_CUDA=1 SWEEP_CUDA_ARCH_LIST="7.0;8.0" TORCH_CUDA_ARCH_LIST="7.0;8.0"
${PY} setup.py build_ext --inplace 2>&1 | tail -2
ls src/sweep/_C*.so || { echo BUILD_FAIL; exit 1; }; echo BUILD_OK

echo; echo "############ correctness re-check (serial path, with fix) ############"
for fam in acoustic elastic; do for nd in 2 3; do
  echo "--- $fam ${nd}D x-cut px2 ---"
  ${TR} --nproc-per-node=2 test/dd_api_check.py --family $fam --ndim $nd --px 2 2>&1 | grep -E "DD_API_CHECK" | tail -1
done; done

echo; echo "############ WEAK 3D 256^3/GPU px8: floor(none) vs serial(full) ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 2>&1 | grep DD_BENCH
echo "-- px8 none (compute floor) --"; ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode none 2>&1 | grep DD_BENCH
echo "-- px8 full (serial comm) --";   ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode full 2>&1 | grep DD_BENCH
echo "-- px8 overlap --";              ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --overlap 2>&1 | grep DD_BENCH

echo; echo "############ NCCL env tuning (px8 full) ############"
echo "-- NCCL_P2P_LEVEL=NVL --"; NCCL_P2P_LEVEL=NVL ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode full 2>&1 | grep DD_BENCH
echo "-- NCCL_ALGO=Ring --";     NCCL_ALGO=Ring ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode full 2>&1 | grep DD_BENCH

echo; echo "############ WEAK larger tile 320^3/GPU px8 (less comm fraction) ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 60 2>&1 | grep DD_BENCH
echo "-- px8 full --"; ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 60 --exchange-mode full 2>&1 | grep DD_BENCH

echo; echo "############ STRONG 3D 256x256x1024 px8: none vs full ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 1024 --nt 100 2>&1 | grep DD_BENCH
echo "-- px8 none --"; ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 128 --nt 100 --exchange-mode none 2>&1 | grep DD_BENCH
echo "-- px8 full --"; ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 128 --nt 100 --exchange-mode full 2>&1 | grep DD_BENCH
echo DONE
