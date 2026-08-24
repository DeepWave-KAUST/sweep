#!/usr/bin/env bash
#SBATCH --job-name=dd-ycut-fix
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-padopt/_dd_cuda/dd-ycut-fix-%j.log

# Verify the py>=3 boundary-saving capture-crash fix (cut_face_mask set in
# DDPropagator._capture). Pure-Python fix -> reuses the existing cut-aware .so,
# NO rebuild. Each case: DDPropagator vs single-domain reference, record +
# per-tile gradient, bitwise (PASS) / rel<=1e-5 (PASS_TOL) / FAIL.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PY} -m torch.distributed.run --standalone"
cd "${WORK}"
export PYTHONPATH="${WORK}/src"
echo "=== host ==="; hostname; nvidia-smi -L | head -1
echo "=== sweep + fix present? ==="
${PY} -c "import sys; sys.path.insert(0,'${WORK}/src'); import sweep; print('sweep', sweep.__file__)"
grep -n 'p.cut_face_mask = self.cut_mask' src/sweep/parallel/dd_propagator.py | head -2
C=test/dd_api_check.py

run() { echo; echo "@@@@@@@@@@ $2 px=$3 py=$4 ${5:-} @@@@@@@@@@"; ${TR} --nproc-per-node="$1" ${C} --family "$2" --ndim 3 --px "$3" --py "$4" ${5:-}; echo "exit=$?"; }

# ---- the two cases the fix unblocks (py>=3: interior rank both y-faces cut) ----
run 8 acoustic 2 4              # px=2 py=4  (the original crashing repro; ny=20)
run 8 elastic  2 4
run 8 acoustic 1 8 "--ny 24"   # px=1 py=8  (pure y-cut, py=8; ny must be %8==0)
run 8 elastic  1 8 "--ny 24"

# ---- regression: the previously-safe balanced py<=2 grids must still PASS ----
run 8 acoustic 4 2             # px=4 py=2
run 8 elastic  4 2

echo; echo "ALL_DONE"
