#!/bin/bash
# KBevo SFT training — canonical wrapper for the paper's two-phase HotpotQA
# supervision on Qwen3-1.7B and Qwen3-4B.
#
# Scheduler-agnostic. Run directly with bash on a local workstation, a cloud
# GPU node, or an already-allocated SLURM node:
#
#   bash scripts/train_sft.sh --model_size 1.7B --dataset_path <sft_json>
#
# To submit to SLURM, use the optional wrapper `scripts/submit_slurm.sh` (see
# README "Optional: Running with Slurm") — it sources configs/cluster.env and
# translates resource variables into sbatch flags.
#
# Args:
#   --model_size {1.7B|4B}            (required)
#   --dataset_path <path>             HotpotQA SFT JSON (else uses KBEVO_SFT_DATA)
#   --debug                           REAL smoke: 1 optimizer step over the
#                                     3-example examples/mini_sft.json into
#                                     an isolated per-job output dir
#   --max_seq_length <int>            override the per-model default
#
# Env (all optional; see configs/cluster.env.example):
#   KBEVO_ENV=kbevo               conda env
#   KBEVO_CKPT_ROOT=<path>        checkpoint root (default $HOME/kbevo_ckpts)
#   KBEVO_SFT_DATA=<path>         default SFT dataset path
#   WANDB_ENTITY/PROJECT          both unset ⇒ W&B disabled
#
# Advisory #SBATCH directives (documentation only; ignored when run under bash):
#SBATCH -J kbevo_sft
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH --requeue

set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/train_sft.sh" ]]; then
    KBEVO_ROOT="$SLURM_SUBMIT_DIR"
else
    KBEVO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
fi
cd "$KBEVO_ROOT"
[[ -f configs/cluster.env ]] && source configs/cluster.env

# ── Conda activation (portable) ──────────────────────────────────────────────
_kbevo_env="${KBEVO_ENV:-kbevo}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$_kbevo_env" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found on PATH — install Miniconda/Anaconda or set KBEVO_ENV to an already-active env." >&2
        exit 3
    fi
    _conda_base="$(conda info --base 2>/dev/null)"
    if [[ -z "$_conda_base" || ! -f "$_conda_base/etc/profile.d/conda.sh" ]]; then
        echo "ERROR: could not locate conda.sh via 'conda info --base'." >&2
        exit 3
    fi
    # shellcheck disable=SC1091
    source "$_conda_base/etc/profile.d/conda.sh"
    if ! conda activate "$_kbevo_env" 2>/dev/null; then
        echo "ERROR: conda env '$_kbevo_env' does not exist. Create it with:" >&2
        echo "  conda env create -f environment.yml" >&2
        exit 3
    fi
fi

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -z "${WANDB_ENTITY:-}" && -z "${WANDB_PROJECT:-}" ]]; then
    export WANDB_MODE=disabled
fi
export MASTER_PORT=$((29501 + RANDOM % 1000))
export NCCL_TIMEOUT=18000
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_USE_CUDA_DSA=1

# ── Defaults ──────────────────────────────────────────────────────────────────
OUTPUT_ROOT="${KBEVO_CKPT_ROOT:-$HOME/kbevo_ckpts}/sft"
MAX_SEQ_LENGTH=""
DEBUG=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)     MODEL_SIZE="$2";      shift 2 ;;
        --dataset_path)   DATASET_PATH="$2";    shift 2 ;;
        --max_seq_length) MAX_SEQ_LENGTH="$2";  shift 2 ;;
        --debug)          DEBUG=1;              shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paper defaults from configs/paper/sft_qwen3_*.yaml (single source of truth) ─
_PAPER_YAML=""
case "${MODEL_SIZE:-}" in
    1.7B) _PAPER_YAML="$KBEVO_ROOT/configs/paper/sft_qwen3_1.7b.yaml"; MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B" ;;
    4B)   _PAPER_YAML="$KBEVO_ROOT/configs/paper/sft_qwen3_4b.yaml";   MODEL_NAME_OR_PATH="Qwen/Qwen3-4B"   ;;
    *)
        echo "ERROR: --model_size {1.7B|4B} required (paper release covers only these two sizes)." >&2
        exit 2 ;;
esac
if [[ -f "$_PAPER_YAML" ]]; then
    eval "$(python "$KBEVO_ROOT/scripts/_paper_config.py" "$_PAPER_YAML")"
fi

# ── Model config (values come from the paper YAML, with hardcoded fallback) ──
NUM_GPUS=1
NUM_TRAIN_EPOCHS=${KBEVO_PAPER__epochs_or_steps__num_train_epochs:-3}
LEARNING_RATE=${KBEVO_PAPER__optimizer__learning_rate:-5e-5}
WARMUP_RATIO=${KBEVO_PAPER__optimizer__warmup_ratio:-0.1}
WEIGHT_DECAY=${KBEVO_PAPER__optimizer__weight_decay:-0.01}
LR_SCHEDULER_TYPE=${KBEVO_PAPER__optimizer__lr_scheduler_type:-cosine}
LOGGING_STEPS=${KBEVO_PAPER__logging_and_checkpointing__logging_steps:-10}
SAVE_STEPS=${KBEVO_PAPER__logging_and_checkpointing__save_steps:-0.125}
SAVE_TOTAL_LIMIT=${KBEVO_PAPER__logging_and_checkpointing__save_total_limit:-8}
EVAL_STRATEGY=${KBEVO_PAPER__logging_and_checkpointing__eval_strategy:-epoch}
PER_DEVICE_TRAIN_BATCH_SIZE=${KBEVO_PAPER__batching__per_device_train_batch_size:-8}
GRADIENT_ACCUMULATION_STEPS=${KBEVO_PAPER__batching__gradient_accumulation_steps:-6}
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-${KBEVO_PAPER__sequence_lengths__max_seq_length:-1024}}"
GRADIENT_CHECKPOINTING="${KBEVO_PAPER__precision__gradient_checkpointing:-false}"

# ── Dataset resolution ────────────────────────────────────────────────────────
# Precedence: --dataset_path > KBEVO_SFT_DATA > (for --debug) examples/mini_sft.json > error.
# The canonical HotpotQA SFT dataset is released on the Hugging Face Hub as
# `kilian-group/KBevo-SFT-hotpotqa-6k`; download `trajectories.json` and pass
# its path here.
if [[ -z "${DATASET_PATH:-}" && -n "${KBEVO_SFT_DATA:-}" ]]; then
    DATASET_PATH="${KBEVO_SFT_DATA}"
fi

if [[ -z "${DATASET_PATH:-}" && -n "${DEBUG}" ]]; then
    DATASET_PATH="${KBEVO_ROOT}/examples/mini_sft.json"
    echo "Debug mode: using bundled examples/mini_sft.json (3 examples)"
fi

if [[ -z "${DATASET_PATH:-}" ]]; then
    echo "ERROR: --dataset_path or KBEVO_SFT_DATA must be set." >&2
    echo "" >&2
    echo "The canonical HotpotQA SFT dataset is released as:" >&2
    echo "  https://huggingface.co/datasets/kilian-group/KBevo-SFT-hotpotqa-6k" >&2
    echo "" >&2
    echo "Download once, then pass the JSON:" >&2
    echo "  huggingface-cli download kilian-group/KBevo-SFT-hotpotqa-6k \\" >&2
    echo "      --repo-type dataset --local-dir <dir>" >&2
    echo "  bash scripts/train_sft.sh --model_size 1.7B \\" >&2
    echo "      --dataset_path <dir>/trajectories.json" >&2
    echo "" >&2
    echo "Schema: {\"examples\": [{...}, ...]} with an annotated_text field on" >&2
    echo "each example (or the (format_version, prompt, answer) trio)." >&2
    exit 2
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: dataset file does not exist: $DATASET_PATH" >&2
    exit 2
fi

# ── Debug: reduce to a single optimization step ─────────────────────────────
# Real smoke: 1 fwd/bwd/optimizer step + save, into a per-invocation isolated
# output dir. Do NOT resume from an earlier debug checkpoint.
if [[ -n "${DEBUG}" ]]; then
    NUM_TRAIN_EPOCHS=1
    LOGGING_STEPS=1
    SAVE_STEPS=1
    SAVE_TOTAL_LIMIT=1
    EVAL_STRATEGY=no
    _MAX_STEPS_ARG="--max_steps 1"
    DEBUG_TAG="debug_${SLURM_JOB_ID:-local}_$(date +%s)"
else
    _MAX_STEPS_ARG=""
fi

# Translate the YAML boolean into the actual accelerate flag.
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
    _GRADIENT_CHECKPOINTING_ARG="--gradient_checkpointing"
else
    _GRADIENT_CHECKPOINTING_ARG=""
fi

# ── Output / run naming ───────────────────────────────────────────────────────
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
export WANDB_NAME="${MODEL_NAME_OR_PATH##*/}-SFT_hotpotqa_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${WANDB_NAME}"
[[ -n "${DEBUG}" ]] && OUTPUT_DIR="${OUTPUT_DIR}-debug/${DEBUG_TAG}"

# ── Resolved configuration ────────────────────────────────────────────────────
echo "════════════════ Resolved SFT configuration ════════════════"
echo "  Model init      : ${MODEL_NAME_OR_PATH}"
echo "  Dataset         : ${DATASET_PATH}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Debug mode      : $([[ -n "$DEBUG" ]] && echo 'ON (real smoke: 1 step)' || echo 'off')"
echo "  ── Hardware ──"
echo "  GPUs            : ${NUM_GPUS}"
echo "  Per-device bsz  : ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  Grad accum      : ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective batch : ${EFFECTIVE_BATCH_SIZE}"
echo "  Max seq length  : ${MAX_SEQ_LENGTH}"
echo "  ── Paper recipe (Appendix B.1) ──"
echo "  Learning rate   : ${LEARNING_RATE}"
echo "  Epochs          : ${NUM_TRAIN_EPOCHS}"
echo "  Warmup ratio    : ${WARMUP_RATIO}"
echo "  Weight decay    : ${WEIGHT_DECAY}"
echo "  LR schedule     : ${LR_SCHEDULER_TYPE}"
echo "  Precision       : bfloat16"
echo "════════════════════════════════════════════════════════════"

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --config_file=configs/accelerate/multi_gpu_${NUM_GPUS}.yaml \
    src/sft_train.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_PATH} \
    --dataset_text_field None \
    --output_dir ${OUTPUT_DIR} \
    --use_special_dblookup_tokens True \
    --plain_baseline False \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    ${_MAX_STEPS_ARG} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --do_train \
    --eval_strategy ${EVAL_STRATEGY} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --save_only_model \
    --logging_steps ${LOGGING_STEPS} \
    --logging_dir ${OUTPUT_DIR}/logs \
    --warmup_ratio ${WARMUP_RATIO} \
    --eval_accumulation_steps 1 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --bf16 True \
    --resume_from_checkpoint ${MODEL_NAME_OR_PATH} \
    ${_GRADIENT_CHECKPOINTING_ARG}
    # paper Table 5: SFT uses "–" (off) for gradient checkpointing; only
    # GRPO enables it. The YAML value flows through here.
