#!/usr/bin/env bash
#SBATCH --job-name=dd-dbg
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-dbg-%j.log

# Full-output acoustic dd_api_check (overlap path) to capture the traceback.
# Reuses the .so already built by the perf job (no rebuild).
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary
export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
TR="${PY} -m torch.distributed.run --standalone"
echo "=== host $(hostname) ==="; ls src/sweep/_C*.so

echo; echo "############ acoustic 2D x-cut px2 (FULL OUTPUT) ############"
${TR} --nproc-per-node=2 test/dd_api_check.py --family acoustic --ndim 2 --px 2 2>&1 | tail -40
echo DONE
