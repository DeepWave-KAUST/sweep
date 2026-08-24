#!/usr/bin/env bash
#SBATCH --job-name=dd-ovl
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-ovl-%j.log

# Validate the TRUE-async comm/compute overlap (no inline req.wait) + measure
# whether it now hides the halo comm. Reuses the .so already built (Python-only
# change). dd_api_check px3 exercises the overlap path (source at nx//2 is in a
# tile interior, away from cuts); px2 hits the serial fallback (source at cut).
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary
export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
TR="${PY} -m torch.distributed.run --standalone"
B=test/dd_nccl_bench.py
echo "=== host $(hostname) ==="; ls src/sweep/_C*.so

echo; echo "############ overlap correctness (px3, src in interior -> overlap) ############"
for nd in 2 3; do
  echo "--- acoustic ${nd}D px3 (overlap) ---"
  ${TR} --nproc-per-node=3 test/dd_api_check.py --family acoustic --ndim $nd --px 3 2>&1 | grep -E "DD_API_CHECK|record:" | tail -2
done
echo "############ serial fallback (px2, src at cut) ############"
for nd in 2 3; do
  echo "--- acoustic ${nd}D px2 (serial) ---"
  ${TR} --nproc-per-node=2 test/dd_api_check.py --family acoustic --ndim $nd --px 2 2>&1 | grep DD_API_CHECK | tail -1
done

echo; echo "############ WEAK 3D 256^3/GPU px8: none / full / async-overlap ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 2>&1 | grep DD_BENCH
echo "-- none --";    ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode none 2>&1 | grep DD_BENCH
echo "-- full --";    ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode full 2>&1 | grep DD_BENCH
echo "-- overlap --"; ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --overlap 2>&1 | grep DD_BENCH

echo; echo "############ STRONG 3D 256x256x1024 px8: none / full / async-overlap ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 1024 --nt 100 2>&1 | grep DD_BENCH
echo "-- none --";    ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 128 --nt 100 --exchange-mode none 2>&1 | grep DD_BENCH
echo "-- full --";    ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 128 --nt 100 --exchange-mode full 2>&1 | grep DD_BENCH
echo "-- overlap --"; ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 128 --nt 100 --overlap 2>&1 | grep DD_BENCH
echo DONE
