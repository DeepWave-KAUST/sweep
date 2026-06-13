#!/usr/bin/env bash
#SBATCH --job-name=dd-round11
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-cuda/_dd_cuda/slurm-%j.log

# ELASTIC DD: NCCL correctness (forward + backward, 2D/3D, FS on/off) then
# the weak/strong/parameter scale matrix. Elastic uses the half-step
# protocol (exchange 5 fields/step in 2D, 9 in 3D) and is SERIAL (no overlap
# mode), so each weak block runs none (timing floor) + full (real); the
# weak-scaling efficiency is none/full in the same job.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-cuda
PYTHON_BIN=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PYTHON_BIN} -m torch.distributed.run --standalone"
EB=test/dd_nccl_elastic_bench.py
CF=test/dd_nccl_elastic_check.py
CB=test/dd_nccl_elastic_backward_check.py

cd "${WORK}"
echo "=== host $(hostname) ==="; nvidia-smi -L
export PYTHONPATH="${WORK}/src"

echo "@@@ E0 correctness: elastic NCCL forward + backward (2D/3D, FS on/off)"
# forward: per-rank tile 48 x 28, global nx = 28*px
${TR} --nproc-per-node=2 ${CF} --fast
${TR} --nproc-per-node=4 ${CF} --fast
${TR} --nproc-per-node=8 ${CF} --fast
${TR} --nproc-per-node=4 ${CF} --fast --free-surface
${TR} --nproc-per-node=2 ${CF} --ndim 3 --fast
${TR} --nproc-per-node=4 ${CF} --ndim 3 --fast
${TR} --nproc-per-node=2 ${CF} --ndim 3 --fast --free-surface
# backward: fixed correctness geometry (2D NX=56, 3D NX=32) split across ranks
${TR} --nproc-per-node=2 ${CB}
${TR} --nproc-per-node=4 ${CB}
${TR} --nproc-per-node=2 ${CB} --free-surface
${TR} --nproc-per-node=2 ${CB} --ndim 3
${TR} --nproc-per-node=2 ${CB} --ndim 3 --free-surface

echo "@@@ E1 WEAK 2D 4096^2/GPU (so4 abcn20 nt400): baselines GPU0/GPU1, px=2/4/8"
for g in 0 1; do CUDA_VISIBLE_DEVICES=${g} ${TR} --nproc-per-node=1 ${EB} --nz 4096 --nxp 4096 --nt 400; done
for p in 2 4 8; do
  ${TR} --nproc-per-node=${p} ${EB} --nz 4096 --nxp 4096 --nt 400 --exchange-mode none
  ${TR} --nproc-per-node=${p} ${EB} --nz 4096 --nxp 4096 --nt 400 --exchange-mode full
done

echo "@@@ E2 WEAK 3D 256^3/GPU (so4 abcn20 nt100): baselines GPU0/GPU1, px=2/4/8"
for g in 0 1; do CUDA_VISIBLE_DEVICES=${g} ${TR} --nproc-per-node=1 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100; done
for p in 2 4 8; do
  ${TR} --nproc-per-node=${p} ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode none
  ${TR} --nproc-per-node=${p} ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --exchange-mode full
done

echo "@@@ E3 STRONG 2D global 4096x16384 (so4 abcn20 nt400): 1/2/4/8 GPUs"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${EB} --nz 4096 --nxp 16384 --nt 400
${TR} --nproc-per-node=2 ${EB} --nz 4096 --nxp 8192 --nt 400 --fast
${TR} --nproc-per-node=4 ${EB} --nz 4096 --nxp 4096 --nt 400 --fast
${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 2048 --nt 400 --fast

echo "@@@ E4 STRONG 3D global 256x256x1024 (so4 abcn20 nt100): 1/2/4/8 GPUs"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 1024 --nt 100
${TR} --nproc-per-node=2 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 512 --nt 100 --fast
${TR} --nproc-per-node=4 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --fast
${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 128 --nt 100 --fast

echo "@@@ E5 SPATIAL ORDER sweep at px=8 weak (so 2 and 8; so4 covered in E1/E2)"
for so in 2 8; do
  ${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt 400 --so ${so} --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt 400 --so ${so} --exchange-mode full
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --so ${so} --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --so ${so} --exchange-mode full
done

echo "@@@ E6 ABCN sweep at px=8 weak (abcn 10 and 40; 20 covered in E1/E2)"
for ab in 10 40; do
  ${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt 400 --abcn ${ab} --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt 400 --abcn ${ab} --exchange-mode full
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --abcn ${ab} --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --abcn ${ab} --exchange-mode full
done

echo "@@@ E7 TILE SIZE sweep at px=8 weak (2D 2048^2/8192^2; 3D 192^3/320^3)"
for s in 2048 8192; do
  ${TR} --nproc-per-node=8 ${EB} --nz ${s} --nxp ${s} --nt 400 --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --nz ${s} --nxp ${s} --nt 400 --exchange-mode full
done
for s in 192 320; do
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz ${s} --ny ${s} --nxp ${s} --nt 100 --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz ${s} --ny ${s} --nxp ${s} --nt 100 --exchange-mode full
done

echo "@@@ E8 NT sweep at px=8 weak (2D nt 100/1600; 3D nt 50/400)"
for n in 100 1600; do
  ${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt ${n} --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt ${n} --exchange-mode full
done
for n in 50 400; do
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt ${n} --exchange-mode none
  ${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt ${n} --exchange-mode full
done

echo "@@@ E9 FREE SURFACE weak px=8 (2D 4096^2, 3D 256^3): FS must not change eff"
${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt 400 --free-surface --exchange-mode none
${TR} --nproc-per-node=8 ${EB} --nz 4096 --nxp 4096 --nt 400 --free-surface --exchange-mode full
${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --free-surface --exchange-mode none
${TR} --nproc-per-node=8 ${EB} --ndim 3 --nz 256 --ny 256 --nxp 256 --nt 100 --free-surface --exchange-mode full

echo "DONE"
