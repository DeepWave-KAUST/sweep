#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

NP="${NP:-4}"
OMP_THREADS="${OMP_THREADS:-14}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MPI_FORWARD_BATCHSIZE="${MPI_FORWARD_BATCHSIZE:-4}"
JOB_NAME="${JOB_NAME:-sweep-cpu-fwi}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/slurm_logs}"
SBATCH_SCRIPT="${SCRIPT_DIR}/cpu_mpi_fwi.sbatch"

mkdir -p "${LOG_DIR}"

SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --time="${TIME_LIMIT}"
    --ntasks="${NP}"
    --ntasks-per-node="${NP}"
    --cpus-per-task="${OMP_THREADS}"
    --chdir="${REPO_ROOT}"
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.out"
)

if [[ -n "${MEM:-}" ]]; then
    SBATCH_ARGS+=(--mem="${MEM}")
fi

if [[ -n "${CONSTRAINT:-}" ]]; then
    SBATCH_ARGS+=(--constraint="${CONSTRAINT}")
fi

echo "Submitting ${JOB_NAME}"
echo "  ranks: ${NP}"
echo "  cpus-per-task: ${OMP_THREADS}"
echo "  time: ${TIME_LIMIT}"
echo "  log dir: ${LOG_DIR}"
echo

JOB_ID_RAW="$(
    MPI_FORWARD_BATCHSIZE="${MPI_FORWARD_BATCHSIZE}" \
    sbatch --parsable "${SBATCH_ARGS[@]}" "${SBATCH_SCRIPT}" "$@"
)"
JOB_ID="${JOB_ID_RAW%%.*}"
LOG_FILE="${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"

echo "Submitted job ${JOB_ID}"
echo "Log file: ${LOG_FILE}"
echo "Waiting for Slurm to create the log..."

while [[ ! -f "${LOG_FILE}" ]]; do
    if ! squeue -j "${JOB_ID}" -h >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if [[ -f "${LOG_FILE}" ]]; then
    echo
    echo "Streaming progress. Press Ctrl-C to stop watching; the job will keep running."
    echo
    tail -n +1 -f "${LOG_FILE}" &
    TAIL_PID=$!
    trap 'kill "${TAIL_PID}" >/dev/null 2>&1 || true; exit 130' INT

    while squeue -j "${JOB_ID}" -h >/dev/null 2>&1; do
        sleep 5
    done

    sleep 2
    kill "${TAIL_PID}" >/dev/null 2>&1 || true
    wait "${TAIL_PID}" >/dev/null 2>&1 || true
else
    echo "Log file was not created yet. Check with: squeue -j ${JOB_ID}"
fi

echo
echo "Job ${JOB_ID} finished or left the queue."
if command -v sacct >/dev/null 2>&1; then
    sacct -j "${JOB_ID}" --format=JobID,JobName,State,Elapsed,MaxRSS,AllocCPUS
fi
