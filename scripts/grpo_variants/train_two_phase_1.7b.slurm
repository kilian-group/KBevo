#!/bin/bash
# KBevo — paper-canonical two-phase GRPO run on Qwen3-1.7B.
#
# Thin SLURM wrapper around `scripts/train_grpo.sh --model_size 1.7B`. The
# canonical implementation, and all paper hyperparameters, live in
# `scripts/train_grpo.sh`.
#
# Recipe (paper §5, Appendix B.1):
#   Init            : KBEVO_SFT_1_7B_CKPT (released as kilian-group/KBevo-Qwen3-1.7B-SFT).
#   Two-phase       : on   (--two_phase)
#   Inverses        : on   (--use_inverses)
#   Retrieval       : top-k=4, threshold=0.6
#   Rollouts        : N=32 QA + K=4 KB per question   → M = N/K = 8 QA per KB
#                     Total generations per question  = K + N = 36
#   Optimizer batch : TOTAL_BATCH_SIZE = 512 (paper effective batch)
#   Steps           : 500
#
# Configure once (see configs/cluster.env.example):
#   KBEVO_SFT_1_7B_CKPT   path to the 1.7B SFT init
#   KBEVO_CKPT_ROOT       where GRPO writes checkpoints
#   KBEVO_ENV             conda env name
#   WANDB_ENTITY/PROJECT  (optional) W&B destination
#
# Submit (partition / GPU are set at submit time — #SBATCH cannot expand vars):
#   sbatch --partition=<your_partition> --gres=gpu:4 \
#          scripts/grpo_variants/train_two_phase_1.7b.slurm
#
# Foreground (for local debug on a 4-GPU node):
#   bash scripts/grpo_variants/train_two_phase_1.7b.slurm --debug

#SBATCH -J kbevo_grpo_1.7b_2ph
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --requeue

set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/train_grpo.sh" ]]; then
    KBEVO_ROOT="$SLURM_SUBMIT_DIR"
else
    KBEVO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
fi
cd "$KBEVO_ROOT"

exec bash "$KBEVO_ROOT/scripts/train_grpo.sh" --model_size 1.7B "$@"
