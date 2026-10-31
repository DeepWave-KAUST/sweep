#!/usr/bin/env bash
# Build the wheel's prebuilt cores -- CUDA 12, CUDA 13 and the host (CPU) core
# -- and the wheel itself inside PyTorch's manylinux_2_28 builder images, so
# the shipped .so needs only glibc >= 2.28 and a 2018-era libstdc++
# (GLIBCXX_3.4.22), exactly like torch's own wheels.  A core linked on a
# developer box binds to that box's libstdc++ (a GLIBCXX version users on
# Debian 11 / RHEL 9 do not have) -- this script exists so that never happens
# again.  Needs docker; ~7 min; no GPU.
#
#   utils/build_cores_manylinux.sh            # cu12 + cu13 + cpu cores, then the wheel
#   utils/build_cores_manylinux.sh cu12       # one core only (no wheel): cu12, cu13 or cpu
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH=${SWEEP_ML_SCRATCH:-$HOME/.cache/sweep-manylinux}   # must be under $HOME for snap docker
IMG12=pytorch/manylinux2_28-builder:cuda12.9
IMG13=pytorch/manylinux2_28-builder:cuda13.0
PY=/opt/python/cp312-cp312/bin/python
run() {  # run <image> <scratch-subdir> <command...>
    local img=$1 sub=$2; shift 2; mkdir -p "$SCRATCH/$sub"
    docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT:/work" -v "$SCRATCH/$sub:/scratch" -w /work \
        -e HOME=/scratch -e TORCH_EXTENSIONS_DIR=/scratch/ext -e PYTHONPATH=/work/src "$img" "$@"
}
want=${1:-all}
if [ "$want" = all ] || [ "$want" = cu12 ]; then
    run $IMG12 cu12 $PY -m sweep.build --core --archs "7.0;7.5;8.0;8.6;8.9;9.0+PTX" --out /work/src/sweep/lib/cu12
fi
if [ "$want" = all ] || [ "$want" = cu13 ]; then
    run $IMG13 cu13 $PY -m sweep.build --core --cuda-home /usr/local/cuda --archs "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX" --out /work/src/sweep/lib/cu13
fi
if [ "$want" = all ] || [ "$want" = cpu ]; then
    # The host (CPU) core: the image's g++ with OpenMP, no toolkit.
    run $IMG12 cpu $PY -m sweep.build --cpu --core --out /work/src/sweep/lib/cpu
fi
newest() { { objdump -T "$1" | grep -oE "\b$2_[0-9.]+" | sort -t_ -k2 -V | tail -1; } || true; }  # <so> <prefix>
for t in cu12 cu13 cpu; do
    so=$ROOT/src/sweep/lib/$t/libsweep_core.so; [ -f "$so" ] || continue
    omp=""
    if [ $t = cpu ]; then
        omp="  $(newest "$so" GOMP) $(newest "$so" OMP)"
        # Inside torch the core binds to torch's own libgomp.so.1 (one OpenMP
        # runtime): it must not need a symbol version newer than that copy's
        # (GOMP_5.0 / OMP_5.0 in torch 2.9's), or dlopen fails next to torch.
        for v in "$(newest "$so" GOMP)" "$(newest "$so" OMP)"; do
            if [ -n "$v" ] && [ "$(printf '%s\n' "${v#*_}" 5.0 | sort -V | tail -1)" != 5.0 ]; then
                echo "cpu: needs $v, newer than torch's libgomp defines (5.0)" >&2; exit 1
            fi
        done
    fi
    echo "$t: GLIBC $(newest "$so" GLIBC)  GLIBCXX $(newest "$so" GLIBCXX)$omp  $(du -h "$so" | cut -f1)"
done
if [ "$want" = all ]; then
    rm -rf "$ROOT/dist" "$ROOT/build" "$ROOT"/src/*.egg-info
    docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT:/work" -w /work -e HOME=/tmp -e SWEEP_REQUIRE_CORE=cu12,cu13,cpu $IMG12 $PY -m build --wheel --no-isolation
    rm -rf "$ROOT/build" "$ROOT"/src/*.egg-info
    ls -la "$ROOT"/dist/*.whl
fi
