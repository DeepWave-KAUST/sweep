#!/usr/bin/env bash
#SBATCH --job-name=dd-round10
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# Comprehensive scale matrix: 1/2/4/8 GPUs, weak+strong, so/abcn/size/nt
# sweeps, 2D+3D. Every weak block carries its own none-mode denominator.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PYTHON_BIN} -m torch.distributed.run --standalone"
B=test/dd_nccl_bench.py

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "@@@ A0 correctness: 8-rank NCCL checks (first time on 8 GPUs)"
${TR} --nproc-per-node=8 test/dd_nccl_check.py --abcn 8 --fast
${TR} --nproc-per-node=8 test/dd_nccl_check.py --ndim 3 --nz 48 --ny 40 --nx 224 --nt 100 --abcn 8 --fast
${TR} --nproc-per-node=8 test/dd_nccl_backward_check.py

echo "@@@ A1 WEAK 2D 4096^2/GPU (so4 abcn20 nt400): baselines GPU0/GPU1, then px=2/4/8"
for g in 0 1; do CUDA_VISIBLE_DEVICES=${g} ${TR} --nproc-per-node=1 ${B} --nz 4096 --nxp 4096 --nt 400; done
for p in 2 4 8; do
  ${TR} --nproc-per-node=${p} ${B} --nz 4096 --nxp 4096 --nt 400 --exchange-mode none
  ${TR} --nproc-per-node=${p} ${B} --nz 4096 --nxp 4096 --nt 400 --exchange-mode full
  ${TR} --nproc-per-node=${p} ${B} --nz 4096 --nxp 4096 --nt 400 --overlap
done

echo "@@@ A2 WEAK 3D 320^3/GPU (so4 abcn20 nt150)"
for g in 0 1; do CUDA_VISIBLE_DEVICES=${g} ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150; done
for p in 2 4 8; do
  ${TR} --nproc-per-node=${p} ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --exchange-mode none
  ${TR} --nproc-per-node=${p} ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --exchange-mode full
  ${TR} --nproc-per-node=${p} ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --overlap
done

echo "@@@ A3 STRONG 2D global 4096x16384 (so4 abcn20 nt400): 1/2/4/8 GPUs"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --nz 4096 --nxp 16384 --nt 400
${TR} --nproc-per-node=2 ${B} --nz 4096 --nxp 8192 --nt 400 --overlap
${TR} --nproc-per-node=4 ${B} --nz 4096 --nxp 4096 --nt 400 --overlap
${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 2048 --nt 400 --overlap

echo "@@@ A4 STRONG 3D global 320x320x1280 (so4 abcn20 nt150): 1/2/4/8 GPUs"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${B} --ndim 3 --nz 320 --ny 320 --nxp 1280 --nt 150
${TR} --nproc-per-node=2 ${B} --ndim 3 --nz 320 --ny 320 --nxp 640 --nt 150 --overlap
${TR} --nproc-per-node=4 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --overlap
${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 160 --nt 150 --overlap

echo "@@@ B SPATIAL ORDER sweep at px=8 weak (so 2 and 8; so4 covered in A)"
for so in 2 8; do
  ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt 400 --so ${so} --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt 400 --so ${so} --overlap
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --so ${so} --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --so ${so} --overlap
done

echo "@@@ C ABCN sweep at px=8 weak (abcn 10 and 40; 20 covered in A)"
for ab in 10 40; do
  ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt 400 --abcn ${ab} --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt 400 --abcn ${ab} --overlap
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --abcn ${ab} --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt 150 --abcn ${ab} --overlap
done

echo "@@@ D TILE SIZE sweep at px=8 weak (2D 2048^2/8192^2; 3D 192^3/448^3)"
for s in 2048 8192; do
  ${TR} --nproc-per-node=8 ${B} --nz ${s} --nxp ${s} --nt 400 --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --nz ${s} --nxp ${s} --nt 400 --overlap
done
for s in 192 448; do
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz ${s} --ny ${s} --nxp ${s} --nt 150 --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz ${s} --ny ${s} --nxp ${s} --nt 150 --overlap
done

echo "@@@ E NT sweep at px=8 weak (2D nt 100/1600; 3D nt 50/600)"
for n in 100 1600; do
  ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt ${n} --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --nz 4096 --nxp 4096 --nt ${n} --overlap
done
for n in 50 600; do
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt ${n} --exchange-mode none
  ${TR} --nproc-per-node=8 ${B} --ndim 3 --nz 320 --ny 320 --nxp 320 --nt ${n} --overlap
done

echo "DONE"
