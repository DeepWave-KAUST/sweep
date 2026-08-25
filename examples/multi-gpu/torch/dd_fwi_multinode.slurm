#!/usr/bin/env bash
# Domain decomposition across NODES, not just across GPUs.
#
# Nothing in the Python changes -- dd_fwi_*.py read LOCAL_RANK and the process
# group, both of which torchrun sets identically whether the ranks sit on one
# machine or twelve.  All this file does is start one torchrun per node.
#
#   sbatch examples/multi-gpu/torch/dd_fwi_multinode.slurm
#   SCRIPT=... ARGS=... sbatch examples/multi-gpu/torch/dd_fwi_multinode.slurm
#
# Not on SLURM?  See "Across nodes" in README.md -- torchrun needs four numbers
# and this script only exists to fill them in from the scheduler.

#SBATCH --job-name=dd-multinode
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=dd-multinode-%j.log

# --ntasks-per-node=1 above is deliberate and easy to get wrong: srun starts ONE
# torchrun per node and torchrun forks the per-GPU workers itself.  Asking for
# one task per GPU starts four torchruns per node, each believing it owns the
# node, and the run deadlocks in rendezvous.  Keep it in step with --gres.
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

SCRIPT=${SCRIPT:-examples/multi-gpu/torch/dd_fwi_overthrust_update.py}
# py * px must equal SLURM_NNODES * GPUS_PER_NODE
ARGS=${ARGS:---py 4 --px 3}

# SCRIPT is relative to the repo root, so submit from there.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=${MASTER_PORT:-29500}

echo "nodes=${nodes[*]}  world=$((SLURM_NNODES * GPUS_PER_NODE))  script=$SCRIPT $ARGS"

# c10d rendezvous: every node runs the SAME command and the ranks negotiate
# their own numbering, so no --node-rank and no scheduler variable reaches
# torchrun.  Swap in --node-rank=$SLURM_NODEID --master-addr/--master-port if
# you need the static form instead; both are verified equivalent here.
#
# Do not pipe this through grep/tail: a pipe swallows the worker traceback and
# a failed run then reports nothing but the launcher's own exit code.
srun --export=ALL --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id="$SLURM_JOB_ID" \
    $SCRIPT $ARGS
