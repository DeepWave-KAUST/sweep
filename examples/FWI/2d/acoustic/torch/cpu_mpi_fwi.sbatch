#!/usr/bin/env bash
#SBATCH --job-name=sweep-cpu-fwi
#SBATCH --time=08:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/FWI/2d/acoustic/torch/fwi_marmousi_cpu_mpi.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
fi
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "/ibex/user/wangs0j/mambaforge/bin/python" ]]; then
        PYTHON_BIN="/ibex/user/wangs0j/mambaforge/bin/python"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Could not find an executable python. Set PYTHON=/path/to/python." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-14}}"
unset MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

MPI_FORWARD_BATCHSIZE="${MPI_FORWARD_BATCHSIZE:-4}"

echo "Job id: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-unknown}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MPI_FORWARD_BATCHSIZE: ${MPI_FORWARD_BATCHSIZE}"
echo

"${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch
import sweep
import sweep._C

threads = int(os.environ.get("OMP_NUM_THREADS", "1").split(",", 1)[0])
torch.set_num_threads(max(1, threads))
torch.set_num_interop_threads(1)

print("Python executable:", sys.executable, flush=True)
print("sweep package:", sweep.__file__, flush=True)
print("sweep extension:", sweep._C.__file__, flush=True)
print("torch num threads:", torch.get_num_threads(), flush=True)
print("torch interop threads:", torch.get_num_interop_threads(), flush=True)
print("OpenMP env OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"), flush=True)
print("OpenMP env MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"), flush=True)
print(flush=True)
if torch.get_num_threads() != threads:
    raise SystemExit(f"Expected torch num threads {threads}, got {torch.get_num_threads()}.")
PY

exec srun \
    --cpu-bind=cores \
    -n "${SLURM_NTASKS:-4}" \
    -c "${SLURM_CPUS_PER_TASK:-14}" \
    "${PYTHON_BIN}" -u "${REPO_ROOT}/examples/FWI/2d/acoustic/torch/fwi_marmousi_cpu_mpi.py" \
    --impl c \
    --mpi \
    --mpi-forward-batchsize "${MPI_FORWARD_BATCHSIZE}" \
    "$@"
