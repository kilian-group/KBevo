#!/bin/bash
# KBevo ablation — curriculum / tier-filtered GRPO.
#
# Compared to the canonical two-phase run, this variant:
#   * Trains on the full HotpotQA training set (train_size=90447), not the
#     6k subset used by the paper's main run.
#   * Loads per-example rollout-success "tier" scores from KBEVO_TIER_PATH
#     and keeps only examples whose score is in [tier_min, tier_max].
#   * Uses --curriculum to expand the tier window over training:
#         phases: 2-3 → 1-3   (2 stages)
#         switch  at 50% of max_steps
#   * Disables --use_inverses.
#   * Initializes from the released 1.7B SFT checkpoint (KBEVO_SFT_1_7B_CKPT).
#   * Keeps --two_phase, --retrieval_threshold 0.6, --retrieval_top_k 4,
#     N=32, K=4, TOTAL_BATCH_SIZE=512 from the canonical recipe.
#
# This is a paper ablation — NOT a released checkpoint.
#
# REQUIRED private artifact:
#   KBEVO_TIER_PATH — a JSON of per-example rollout tier scores over the full
#                     HotpotQA train set. NOT part of the paper's public
#                     release; see configs/cluster.env.example.
#
# Submit (partition / GPU are set at submit time — #SBATCH cannot expand vars):
#   sbatch --partition=<your_partition> --gres=gpu:1 \
#          scripts/grpo_variants/train_curriculum.slurm

#SBATCH -J kbevo_grpo_1.7b_curriculum
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
[[ -f configs/cluster.env ]] && source configs/cluster.env

: "${KBEVO_TIER_PATH:?Set KBEVO_TIER_PATH to the per-example tier-score JSON (private artifact — see configs/cluster.env.example)}"

exec bash "$KBEVO_ROOT/scripts/train_grpo.sh" \
    --model_size 1.7B \
    --no_use_inverses \
    --train_size 90447 \
    --tier_path "$KBEVO_TIER_PATH" \
    --tier_min_score 1 \
    --tier_max_score 3 \
    --curriculum \
    --curriculum_phases "2-3,1-3" \
    --curriculum_steps "0.5" \
    "$@"
