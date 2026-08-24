#!/usr/bin/env bash
#SBATCH --job-name=dd-axisgen
#SBATCH --partition=batch
#SBATCH --gres=gpu:v100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/wangs0j/sweep-dd-padopt/_dd_cuda/slurm-axisgen-%j.log

# Does "balanced 2-D cut beats 1-D x-cut for strong scaling" GENERALIZE across
# global shapes? (256x256x1024 showed px4py2 7.18x vs x-cut 6.55x.) Test two
# more globals at 8 GPUs. px1 ref only where it fits in 32 GB V100.
set -uo pipefail
WORK=/ibex/user/wangs0j/sweep-dd-padopt
PY=/ibex/user/wangs0j/mambaforge/bin/python
TR="${PY} -m torch.distributed.run --standalone"
cd "${WORK}"
export PYTHONPATH="${WORK}/src"
echo "=== host $(hostname) ==="; nvidia-smi -L | head -1
C=test/dd_ddp_timing.py
NT=200

echo "########## GLOBAL 384x384x384 (cubic) ##########"
echo "@@@ px1 ref"
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc-per-node=1 ${C} --px 1 --py 1 --nz 384 --ny 384 --nx 384 --nt ${NT}
echo "@@@ x-cut px8 py1  tile(384,384,48)"
${TR} --nproc-per-node=8 ${C} --px 8 --py 1 --nz 384 --ny 384 --nx 384 --nt ${NT}
echo "@@@ balanced px4 py2  tile(384,192,96)"
${TR} --nproc-per-node=8 ${C} --px 4 --py 2 --nz 384 --ny 384 --nx 384 --nt ${NT}
echo "@@@ balanced px2 py4  tile(384,96,192)"
${TR} --nproc-per-node=8 ${C} --px 2 --py 4 --nz 384 --ny 384 --nx 384 --nt ${NT}

echo "########## GLOBAL 512x512x512 (cubic, px1 OOMs on V100 -> ratio only) ##########"
echo "@@@ x-cut px8 py1  tile(512,512,64)"
${TR} --nproc-per-node=8 ${C} --px 8 --py 1 --nz 512 --ny 512 --nx 512 --nt ${NT}
echo "@@@ balanced px4 py2  tile(512,256,128)"
${TR} --nproc-per-node=8 ${C} --px 4 --py 2 --nz 512 --ny 512 --nx 512 --nt ${NT}
echo "@@@ balanced px2 py4  tile(512,128,256)"
${TR} --nproc-per-node=8 ${C} --px 2 --py 4 --nz 512 --ny 512 --nx 512 --nt ${NT}

echo "########## GLOBAL 256x512x1024 (anisotropic; big y) ##########"
echo "@@@ x-cut px8 py1  tile(256,512,128)"
${TR} --nproc-per-node=8 ${C} --px 8 --py 1 --nz 256 --ny 512 --nx 1024 --nt ${NT}
echo "@@@ balanced px4 py2  tile(256,256,256)"
${TR} --nproc-per-node=8 ${C} --px 4 --py 2 --nz 256 --ny 512 --nx 1024 --nt ${NT}

echo "DONE"
