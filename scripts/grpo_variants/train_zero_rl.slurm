#!/bin/bash
# KBevo ablation — "zero-RL" variant.
#
# Skips the SFT stage entirely and runs GRPO directly on the public raw
# Qwen3-1.7B base model, using a chat-template prompt that teaches the KB
# protocol from scratch.
#
# Compared to the canonical two-phase run, this variant:
#   * Uses MODEL_PATH="Qwen/Qwen3-1.7B" (no SFT init).
#   * Enables --use_chat_template.
#   * Sets --phase1_prompt_type formatted_zero_rl_v8 (from-scratch prompt).
#   * Uses --reward_func f1_format (adds format-adherence reward on top of F1).
#   * Disables --use_inverses.
#   * Keeps --two_phase, --retrieval_threshold 0.6, --retrieval_top_k 4,
#     N=32, K=4, TOTAL_BATCH_SIZE=512 from the canonical recipe.
#
# This is a paper ablation — NOT a released checkpoint. The base model
# `Qwen/Qwen3-1.7B` is public on Hugging Face.
#
# Submit (partition / GPU are set at submit time — #SBATCH cannot expand vars):
#   sbatch --partition=<your_partition> --gres=gpu:1 \
#          scripts/grpo_variants/train_zero_rl.slurm

#SBATCH -J kbevo_grpo_1.7b_zero_rl
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
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

exec env MODEL_PATH="Qwen/Qwen3-1.7B" \
    bash "$KBEVO_ROOT/scripts/train_grpo.sh" \
        --no_use_inverses \
        --use_chat_template \
        --phase1_prompt_type formatted_zero_rl_v8 \
        --reward_func f1_format \
        "$@"
