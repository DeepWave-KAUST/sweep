#!/usr/bin/env bash
# Build the wheel's prebuilt CUDA cores and the wheel itself inside PyTorch's
# manylinux_2_28 builder images, so the shipped .so needs only glibc >= 2.28
# and a 2018-era libstdc++ (GLIBCXX_3.4.22), exactly like torch's own wheels.
# A core linked on a developer box binds to that box's libstdc++ (a GLIBCXX
# version users on Debian 11 / RHEL 9 do not have) -- this script exists so
# that never happens again.  Needs docker; ~5 min; no GPU.
#
#   utils/build_cores_manylinux.sh            # cu12 + cu13 cores, then the wheel
#   utils/build_cores_manylinux.sh cu12       # one core only (no wheel)
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
for t in cu12 cu13; do
    so=$ROOT/src/sweep/lib/$t/libsweep_core.so; [ -f "$so" ] || continue
    echo "$t: GLIBC $(objdump -T "$so" | grep -oE 'GLIBC_[0-9.]+' | sort -t_ -k2 -V | tail -1)  GLIBCXX $(objdump -T "$so" | grep -oE 'GLIBCXX_[0-9.]+' | sort -t_ -k2 -V | tail -1)  $(du -h "$so" | cut -f1)"
done
if [ "$want" = all ]; then
    rm -rf "$ROOT/dist" "$ROOT/build" "$ROOT"/src/*.egg-info
    docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT:/work" -w /work -e HOME=/tmp -e SWEEP_REQUIRE_CORE=cu12,cu13 $IMG12 $PY -m build --wheel --no-isolation
    rm -rf "$ROOT/build" "$ROOT"/src/*.egg-info
    ls -la "$ROOT"/dist/*.whl
fi
