#!/usr/bin/env bash
#SBATCH --job-name=dd-fscale
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-fscale-%j.log

# Final scaling curve (committed overlap), acoustic 3D, 1/2/4/8 GPUs.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary
export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
TR="${PY} -m torch.distributed.run --standalone"
B=test/dd_nccl_bench.py
echo "=== host $(hostname) ==="; ls src/sweep/_C*.so

echo; echo "############ WEAK 3D 256^3/GPU (overlap): 1/2/4/8 ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 2>&1 | grep DD_BENCH
for p in 2 4 8; do
  ${TR} --nproc-per-node=${p} ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --overlap 2>&1 | grep DD_BENCH
done

echo; echo "############ STRONG 3D global 256x256x1024 (overlap): 1/2/4/8 ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 1024 --nt 100 2>&1 | grep DD_BENCH
for p in 2 4 8; do
  ${TR} --nproc-per-node=${p} ${B} --ndim 3 --nz 256 --ny 256 --nxp $((1024 / p)) --nt 100 --overlap 2>&1 | grep DD_BENCH
done

echo; echo "############ WEAK larger 320^3/GPU (overlap): 1 vs 8 ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 60 2>&1 | grep DD_BENCH
${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 60 --overlap 2>&1 | grep DD_BENCH
echo DONE
