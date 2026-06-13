#!/usr/bin/env bash
#SBATCH --job-name=padopt-perf
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/ibex/user/wangs0j/_sgbench/padopt-perf-%j.log

# DD performance: confirm cut-aware correctness (fp.cut_face_mask fix +
# DDPropagator acoustic comm/compute overlap), then weak/strong scaling toward
# 8x on 8 GPUs (serial vs SPECFEM overlap).
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
export CUDA_HOME=/sw/rl9g/cuda/12.4.1/rl9_binary
export PATH="${CUDA_HOME}/bin:${PATH}"
cd "${WORK}"; export PYTHONPATH="${WORK}/src"
TR="${PY} -m torch.distributed.run --standalone"
B=test/dd_nccl_bench.py
echo "=== host $(hostname) ==="; nvidia-smi -L

echo; echo "############ BUILD multiarch ############"
rm -rf build
export SWEEP_BUILD_CUDA=1 SWEEP_CUDA_ARCH_LIST="7.0;8.0" TORCH_CUDA_ARCH_LIST="7.0;8.0"
${PY} setup.py build_ext --inplace 2>&1 | tail -3
ls src/sweep/_C*.so || { echo BUILD_FAIL; exit 1; }; echo BUILD_OK

echo; echo "############ correctness: dd_api_check (overlap path, with fix) ############"
for fam in acoustic elastic; do
  for nd in 2 3; do
    echo "--- $fam ${nd}D x-cut px2 ---"
    ${TR} --nproc-per-node=2 test/dd_api_check.py --family $fam --ndim $nd --px 2 2>&1 | grep -E "DD_API_CHECK|FAIL" | tail -1
  done
done
echo "--- acoustic 3D 2x2 px2py2 ---"
${TR} --nproc-per-node=4 test/dd_api_check.py --family acoustic --ndim 3 --px 2 --py 2 2>&1 | grep -E "DD_API_CHECK|FAIL" | tail -1

echo; echo "############ acoustic WEAK 2D 4096^2/GPU (px=8): serial vs overlap ############"
for g in 0 1; do CUDA_VISIBLE_DEVICES=${g} ${TR} --nproc-per-node=1 ${B} --nz 4096 --nxp 4096 --nt 400 2>&1 | grep "DD_BENCH"; done
echo "-- px8 serial --";  ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt 400 --fast 2>&1 | grep "DD_BENCH"
echo "-- px8 overlap --"; ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt 400 --overlap 2>&1 | grep "DD_BENCH"

echo; echo "############ acoustic WEAK 3D 256^3/GPU (px=8): serial vs overlap ############"
for g in 0 1; do CUDA_VISIBLE_DEVICES=${g} ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 2>&1 | grep "DD_BENCH"; done
echo "-- px8 serial --";  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --fast 2>&1 | grep "DD_BENCH"
echo "-- px8 overlap --"; ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --overlap 2>&1 | grep "DD_BENCH"

echo; echo "############ acoustic STRONG 3D global 256x256x1024: 1/2/4/8 (overlap) ############"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 256 --ny 256 --nxp 1024 --nt 100 2>&1 | grep "DD_BENCH"
for p in 2 4 8; do
  echo "-- p=${p} overlap --"; ${TR} --nproc-per-node=${p} ${B} --ndim 3 --nz 256 --ny 256 --nxp $((1024 / p)) --nt 100 --overlap 2>&1 | grep "DD_BENCH"
done

echo; echo "############ memory A/B (cut-aware vs symmetric, single GPU) ############"
CUDA_VISIBLE_DEVICES=0 ${PY} _dd_cuda/mem_ab.py 2>&1 | grep -vE "Warning|warn"
echo DONE
