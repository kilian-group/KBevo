#!/bin/bash
# KBevo ablation — vanilla-GRPO variant on Qwen3-1.7B.
#
# Compared to the canonical two-phase run, this variant:
#   * Enables --vanilla_grpo (uses the standard GRPO advantage/update
#     instead of KBevo's two-phase advantage grouping).
#   * Halves --num_generations to 16 (paper's vanilla-GRPO N).
#   * Disables --use_inverses.
#   * Keeps --two_phase (rollout structure) and everything else at the
#     canonical values (--retrieval_threshold 0.6, --retrieval_top_k 4,
#     K=4, TOTAL_BATCH_SIZE=512).
#   * Initializes from the released 1.7B SFT checkpoint (KBEVO_SFT_1_7B_CKPT).
#
# This is a paper ablation — NOT a released checkpoint.
#
# Submit (partition / GPU are set at submit time — #SBATCH cannot expand vars):
#   sbatch --partition=<your_partition> --gres=gpu:1 \
#          scripts/grpo_variants/train_vanilla_grpo.slurm

#SBATCH -J kbevo_grpo_1.7b_vanilla
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
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

exec bash "$KBEVO_ROOT/scripts/train_grpo.sh" \
    --model_size 1.7B \
    --no_use_inverses \
    --vanilla_grpo \
    --num_generations 16 \
    "$@"
