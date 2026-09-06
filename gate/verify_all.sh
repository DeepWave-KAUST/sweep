#!/usr/bin/env bash
# Everything that protects this branch, in one run.
#
# The bit-exactness gate is the biggest instrument but it is not the only one,
# and it has two structural blind spots that the others exist to cover:
#
#   * it compares C against C, so it cannot see a defect the full and
#     boundary-saving paths SHARE  -> the eager leg of the gradient matrix;
#   * its ALL_SOLVERS omits elastic_vr2d, elastic_tti_sg3d, elastic_tti_2nd2d
#     and visco_acoustic2d, and no tier ever sets compute_illumination
#                                  -> evr_ab.py, the visco suite, the pin test.
#
# Usage:  . gate/env.sh && gate/verify_all.sh [--quick]
#   --quick  skips gate tier B (165 configs, the long pole) and the full matrix.
set -u
cd "$(dirname "$0")/.." || exit 1
QUICK=${1:-}
LOG=${VERIFY_LOG_DIR:-/tmp/sweep-verify-$$}
mkdir -p "$LOG"
: "${PY:?run '. gate/env.sh' first}"
echo "logs -> $LOG"
rc_total=0

run () {   # run <name> <command...>
    local name=$1; shift
    printf '%-34s ' "$name"
    if "$@" > "$LOG/$name.log" 2>&1; then
        printf 'ok\n'
    else
        printf 'FAILED  (see %s/%s.log)\n' "$LOG" "$name"; rc_total=1
    fi
}

echo "=== 1. bit-exactness gate (C vs a stored baseline) ==="
if [ "$QUICK" = "--quick" ]; then
    run gate_ACdd1 gate/run_gate.sh A base_A.pt C base_C.pt dd1 base_dd1.pt
else
    run gate_all gate/run_gate.sh A base_A.pt C base_C.pt B base_B.pt dd1 base_dd1.pt
fi

echo "=== 2. elastic_vr2d -- the one CUDA equation outside every tier ==="
run evr_ab "$PY" gate/evr_ab.py --out "$LOG/evr.pt" --compare gate/ab_evr_post.pt

echo "=== 3. eager vs C -- the only reference that does not share code with"
echo "       the compiled backward, so the only one that sees a shared defect ==="
if [ "$QUICK" = "--quick" ]; then
    run grad_matrix_short env PYTHONPATH=src OMP_NUM_THREADS=8 "$PY" \
        test/backend_gradient_matrix.py --scale cuda-suite --backends eager/gpu c-cuda \
        --cases acoustic2d acoustic3d acoustic_vti_1st_2d acoustic_vti_1st_3d
else
    # --expect is what makes this step able to fail: without it the matrix
    # printed "Failures:" and still exited 0, so this line reported ok while
    # sixteen cells were red.
    run grad_matrix env PYTHONPATH=src OMP_NUM_THREADS=8 "$PY" \
        test/backend_gradient_matrix.py --scale cuda-suite --backends eager/gpu c-cuda \
        --expect gate/expected_matrix_failures.txt
fi

echo "=== 4. the equations the gate does not run ==="
run visco_cuda env OMP_NUM_THREADS=8 "$PY" -m pytest -q -p no:cacheprovider test/test_visco_acoustic_cuda.py
run vti1st_adjoint env OMP_NUM_THREADS=8 "$PY" -m pytest -q -p no:cacheprovider test/test_acoustic_vti_1st_c_adjoint.py

echo "=== 5. the illumination path -- no gate tier ever enables it ==="
run illumination_pin env OMP_NUM_THREADS=8 "$PY" -m pytest -q -p no:cacheprovider test/test_illumination_pin.py

echo
echo "NOT covered by any of the above, as of this writing:"
echo "  * DD beyond ddgate's tiles (Acoustic, Acoustic3D, AcousticVRZ3D, Elastic,"
echo "    Elastic3D) -- elastic staged boundary needs 2 GPUs; see"
echo "    test/dd_elastic_staged_check.py"
echo "  * the source-wavelet gradient of acoustic_vti_1st_2d/3d -- impl='c'"
echo "    returns None; declared in KNOWN_MISSING_GRADIENTS"
echo "  * receiver_illumination's missing it==0 term on the boundary-saving path"
echo "    -- pinned by test_illumination_pin.py, not fixed"
echo
[ $rc_total -eq 0 ] && echo "VERIFY-ALL: ok" || echo "VERIFY-ALL: SOMETHING FAILED"
exit $rc_total
