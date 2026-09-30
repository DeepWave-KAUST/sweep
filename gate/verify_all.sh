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
        --cases acoustic2d acoustic3d acoustic_vti_1st_2d acoustic_vti_1st_3d \
        --expect gate/expected_matrix_failures_quick.txt
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

echo "=== 5. DD against ONE CARD -- ddgate compares DD to a stored DD baseline,"
echo "       whose twelve entries are all 1x1, so it has never seen a cut ==="
# No stored baseline on purpose: a .pt of DD numbers is exactly what let a
# single-domain run wearing a ModelParallel wrapper read as "dd1 12/12" for this
# repository's whole history. The single card IS the reference, computed in the
# same process, and bit-exact is the criterion.
NG=$("$PY" -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
dd_case () {   # dd_case <world> <name> <args...>
    local world=$1 name=$2; shift 2
    if [ "$NG" -lt "$world" ]; then
        printf '%-34s skipped (needs %s GPUs, %s visible)\n' "$name" "$world" "$NG"
        return
    fi
    # ${DD_EXTRA_ENV} goes through env, not as a "VAR=x func" prefix: bash
    # leaves such an assignment set in the shell after a FUNCTION call, so the
    # next case would silently inherit it.
    run "$name" env ${DD_EXTRA_ENV:-} OMP_NUM_THREADS=8 "$PY" -m torch.distributed.run \
        --standalone --nproc-per-node="$world" test/dd_vs_mono_grad_sweep.py "$@"
}
for eq in Acoustic Elastic; do
    dd_case 2 "dd_vs_mono_${eq}_1x2" --equation "$eq" --py 1 --px 2
done
for eq in Acoustic3D Elastic3D; do
    dd_case 2 "dd_vs_mono_${eq}_1x2" --equation "$eq" --py 1 --px 2
    dd_case 4 "dd_vs_mono_${eq}_2x2" --equation "$eq" --py 2 --px 2
done
# n=57 keeps both split axes even so pad_to_mesh is a no-op: with a pad, the
# replicate adjoint folds the pad plane back AFTER the cross-shot sum instead of
# before it, which costs one fp32 ULP on one plane and is not a defect (item 30).
dd_case 4 "dd_vs_mono_shotgroups" --equation Acoustic3D --grid-n 57 --shot-groups 2 \
        --py 1 --px 2
# One card picks the FUSED VRZ gradient kernel and DD must use the SPLIT one;
# same kernel, bit-exact. Without this the arm is red for a reason that is not
# a defect.
DD_EXTRA_ENV=SWEEP_VRZ_GRAD_SPLIT=1 \
    dd_case 2 "dd_vs_mono_VRZ3D_1x2_split" --equation AcousticVRZ3D --py 1 --px 2
unset DD_EXTRA_ENV

echo "=== 6. the illumination path -- no gate tier ever enables it ==="
run illumination_pin env OMP_NUM_THREADS=8 "$PY" -m pytest -q -p no:cacheprovider test/test_illumination_pin.py

echo
echo "NOT covered by any of the above, as of this writing:"
echo "  * DD on more than the visible GPU count -- step 5 skips what it cannot"
echo "    run, and says so; the full py/px/shot-group sweep is a 4-GPU job"
echo "  * elastic staged boundary under DD -- needs 2 GPUs; see"
echo "    test/dd_elastic_staged_check.py"
echo "  * the source-wavelet gradient of acoustic_vti_1st_2d/3d -- impl='c'"
echo "    returns None; declared in KNOWN_MISSING_GRADIENTS"
echo "  * receiver_illumination's missing it==0 term on the boundary-saving path"
echo "    -- pinned by test_illumination_pin.py, not fixed"
echo
[ $rc_total -eq 0 ] && echo "VERIFY-ALL: ok" || echo "VERIFY-ALL: SOMETHING FAILED"
exit $rc_total
