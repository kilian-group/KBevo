#!/bin/bash
# KBevo ablation — "one-phase" GRPO variant.
#
# Compared to the canonical two-phase run, this variant:
#   * Initializes from a 1-phase SFT model (single-response supervision,
#     no phase-1 KB rollouts during SFT).
#   * Disables --two_phase   ⇒  no phase-1 KB construction at rollout time.
#   * Provides a pre-built KB via --database_path (a JSON extracted offline
#     from Gemini) instead of building one per question.
#   * Keeps --use_inverses, --retrieval_threshold 0.6, --retrieval_top_k 4,
#     N=32, K=4, TOTAL_BATCH_SIZE=512 from the canonical recipe.
#
# NOT part of the paper's public checkpoint release. Both inputs below are
# private artifacts:
#   * KBEVO_ONE_PHASE_SFT_1_7B_CKPT — a 1-phase Qwen3-1.7B SFT checkpoint.
#   * KBEVO_ONE_PHASE_DB            — the offline-extracted HotpotQA DB JSON.
# If you do not have these, this variant cannot be reproduced from the
# release alone.
#
# Submit (partition / GPU are set at submit time — #SBATCH cannot expand vars):
#   sbatch --partition=<your_partition> --gres=gpu:4 \
#          scripts/grpo_variants/train_one_phase.slurm

#SBATCH -J kbevo_grpo_1phase
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

: "${KBEVO_ONE_PHASE_SFT_1_7B_CKPT:?Set KBEVO_ONE_PHASE_SFT_1_7B_CKPT to a 1-phase SFT checkpoint (private artifact — see configs/cluster.env.example)}"
: "${KBEVO_ONE_PHASE_DB:?Set KBEVO_ONE_PHASE_DB to the Gemini-extracted HotpotQA database JSON (private artifact — see configs/cluster.env.example)}"

exec env MODEL_PATH="$KBEVO_ONE_PHASE_SFT_1_7B_CKPT" \
    bash "$KBEVO_ROOT/scripts/train_grpo.sh" \
        --no_two_phase \
        --database_path "$KBEVO_ONE_PHASE_DB" \
        "$@"
