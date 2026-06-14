#!/usr/bin/env bash
#SBATCH --job-name=dd-cprof
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/dd-cprof-%j.log
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary; export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
echo "=== host $(hostname) ==="; ls src/sweep/_C*.so
${PY} -m torch.distributed.run --standalone --nproc-per-node=2 _dd_cuda/comm_profile.py 2>&1 | grep -E "\[profile\]|Error|Traceback" | tail -15
echo DONE
