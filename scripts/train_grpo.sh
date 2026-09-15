#!/bin/bash
# KBevo GRPO training — single self-contained script for both 1.7B and 4B.
# Runs the paper-canonical 2-phase recipe:
#   --two_phase --use_inverses --retrieval_threshold 0.6 --retrieval_top_k 4
#   N = --num_generations = 32     (Phase-2 QA rollouts per question)
#   K = --num_db_rollouts  = 4     (Phase-1 KB rollouts per question)
#   M = N / K              = 8     (Phase-2 QA rollouts per (question, KB))
#   Total generations per question = K + N = 36
#   Optimizer batch size (TOTAL_BATCH_SIZE) = 512     — paper effective batch
#   Steps = 500
# The rollout count (36 per question) and the optimizer effective batch (512
# examples per gradient step) are independent knobs.
#
# Foreground:  bash   scripts/train_grpo.sh --model_size 1.7B
# SLURM:       sbatch --partition=<your_partition> --gres=gpu:nvidia_b200:4 scripts/train_grpo.sh --model_size 1.7B
#
# Required args:
#   --model_size {1.7B|4B}
#
# Optional flags:
#   --debug                       short debug run (TRAIN_SIZE=1000, EVAL_SIZE=10)
#   --any-other-body-flag=value   forwarded to the underlying training body
#                                 (e.g. --learning_rate=1e-6, --max_steps=100,
#                                  --num_generations=32, --tier_path=<path>, ...)
#
# Env (see configs/cluster.env.example):
#   KBEVO_SFT_1_7B_CKPT=<path>   1.7B SFT checkpoint to initialize from
#   KBEVO_SFT_4B_CKPT=<path>     4B SFT checkpoint to initialize from
#   KBEVO_CKPT_ROOT=<path>       where GRPO writes checkpoints (default $HOME/kbevo_ckpts)
#   KBEVO_ENV=kbevo              conda env
#   MODEL_PATH=<path>            fully override MODEL_PATH (advanced; skips --model_size)

#SBATCH -J kbevo_grpo
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
    KBEVO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
fi
cd "$KBEVO_ROOT"
[[ -f configs/cluster.env ]] && source configs/cluster.env

# ── Conda activation (portable) ──────────────────────────────────────────────
# Activate ${KBEVO_ENV:-kbevo} unless it's already active. Fresh Bash processes
# do not inherit the `conda` shell function, so we source conda.sh explicitly.
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

# Per-job caches so concurrent shards don't race on /tmp / torch.compile hashes.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/kbevo/triton_${SLURM_JOB_ID:-local}}"
export TMPDIR="${TMPDIR:-$HOME/.cache/kbevo/tmp_${SLURM_JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

export MASTER_PORT=${MASTER_PORT:-$((29501 + RANDOM % 1000))}

# ── Preamble: pull --model_size and --debug off the argv, forward everything else ─────
MODEL_SIZE=""
DEBUG_ARG=""
PRESERVED_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --debug)      DEBUG_ARG="--debug"; shift 1 ;;
        *)            PRESERVED_ARGS+=("$1"); shift 1 ;;
    esac
done

# ── Resolve MODEL_PATH ───────────────────────────────────────────────────────
# Precedence: MODEL_PATH env > KBEVO_SFT_{SIZE}_CKPT > public HF repo default.
# The resolved value may be a local checkpoint dir OR a Hugging Face repo id
# (owner/name). HF ids are materialized locally with huggingface_hub.snapshot_
# download so vLLM / verl can load them from disk.
if [[ -z "${MODEL_PATH:-}" ]]; then
    case "$MODEL_SIZE" in
        1.7B)  MODEL_PATH="${KBEVO_SFT_1_7B_CKPT:-kilian-group/KBevo-Qwen3-1.7B-SFT}" ;
               SAVE_DIR_DEFAULT="${KBEVO_CKPT_ROOT:-$HOME/kbevo_ckpts}/grpo_1.7b" ;;
        4B)    MODEL_PATH="${KBEVO_SFT_4B_CKPT:-kilian-group/KBevo-Qwen3-4B-SFT}"   ;
               SAVE_DIR_DEFAULT="${KBEVO_CKPT_ROOT:-$HOME/kbevo_ckpts}/grpo_4b" ;;
        *)     echo "ERROR: --model_size {1.7B|4B} required (or set MODEL_PATH env)" >&2 ; exit 2 ;;
    esac
else
    SAVE_DIR_DEFAULT="${KBEVO_CKPT_ROOT:-$HOME/kbevo_ckpts}/grpo_custom"
fi

# If MODEL_PATH is not an existing directory, treat it as a Hugging Face repo
# id and snapshot_download it. `python -` reads the id via env, so it is never
# interpolated into source text. Only the last stdout line is captured, so
# hub warnings / progress do not leak into the resolved path.
if [[ ! -d "$MODEL_PATH" ]]; then
    if [[ "$MODEL_PATH" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
        echo "Resolving HF repo '$MODEL_PATH' via huggingface_hub.snapshot_download..." >&2
        _snapshot="$(MODEL_ID="$MODEL_PATH" python - <<'PY' 2>/dev/null | tail -n 1
import os, sys
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ["MODEL_ID"]))
PY
)"
        if [[ -z "$_snapshot" || ! -d "$_snapshot" ]]; then
            echo "ERROR: snapshot_download('$MODEL_PATH') did not resolve to a local directory." >&2
            echo "       Check your network / HF token; or pass a local --model_path." >&2
            exit 2
        fi
        MODEL_PATH="$_snapshot"
    else
        echo "ERROR: MODEL_PATH '$MODEL_PATH' is neither a local directory nor an owner/name HF repo id." >&2
        exit 2
    fi
fi

SAVE_DIR="${SAVE_DIR:-$SAVE_DIR_DEFAULT}"
mkdir -p "$SAVE_DIR"

# ── W&B: unset entity+project ⇒ disable; otherwise preserve user values ─────
if [[ -z "${WANDB_ENTITY:-}" && -z "${WANDB_PROJECT:-}" ]]; then
    export WANDB_MODE=disabled
fi

# ── Load paper YAML into env BEFORE the set -- block, so canonical flags
#    below read their values from the YAML (single source of truth). Any
#    later CLI flag in PRESERVED_ARGS still wins (arg parser runs after).
_PAPER_YAML=""
case "$MODEL_SIZE" in
    1.7B) _PAPER_YAML="$KBEVO_ROOT/configs/paper/grpo_qwen3_1.7b.yaml" ;;
    4B)   _PAPER_YAML="$KBEVO_ROOT/configs/paper/grpo_qwen3_4b.yaml"   ;;
esac
if [[ -n "$_PAPER_YAML" && -f "$_PAPER_YAML" ]]; then
    eval "$(python "$KBEVO_ROOT/scripts/_paper_config.py" "$_PAPER_YAML")"
fi

# Conditional bool flags derived from YAML (bare-flag pattern — omit to keep off).
_TWO_PHASE_FLAG=""
[[ "${KBEVO_PAPER__phase1__two_phase:-true}"        == "true" ]] && _TWO_PHASE_FLAG="--two_phase"
_USE_INVERSES_FLAG=""
[[ "${KBEVO_PAPER__retrieval__use_inverses:-true}"  == "true" ]] && _USE_INVERSES_FLAG="--use_inverses"

# Feed the paper-canonical arg set into the body's argparse (below). Every
# value flows from configs/paper/<yaml>; the hardcoded fallbacks after `:-`
# only kick in if the YAML is deleted / broken.
set -- \
    --gpu_type B200 \
    --model_path "$MODEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --dataset_name hotpotqa \
    --database_path "${KBEVO_PAPER__data__database_path:-}" \
    --train_size ${KBEVO_PAPER__data__train_examples:-7000} \
    --reward_func ${KBEVO_PAPER__reward__reward_func:-f1} \
    --total_batch_size ${KBEVO_PAPER__batching__effective_batch_size:-512} \
    --phase1_prompt_type ${KBEVO_PAPER__phase1__phase1_prompt_type:-sft} \
    ${_TWO_PHASE_FLAG} \
    --retrieval_threshold ${KBEVO_PAPER__retrieval__retrieval_threshold:-0.6} \
    --retrieval_top_k ${KBEVO_PAPER__retrieval__retrieval_top_k:-4} \
    --num_generations ${KBEVO_PAPER__rollouts__N_phase2_qa_per_question:-32} \
    --num_db_rollouts ${KBEVO_PAPER__rollouts__K_phase1_kb_per_question:-4} \
    ${_USE_INVERSES_FLAG} \
    $DEBUG_ARG \
    "${PRESERVED_ARGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# GRPO Training body (formerly scripts/grpo_train.sh)
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
GPU_TYPE=""
MODEL_PATH=""
DATABASE_PATH=""  # Not used in two-phase mode
SAVE_DIR="${KBEVO_CKPT_ROOT:-$HOME/kbevo_ckpts}/grpo_debug"
DATASET_NAME="hotpotqa"
NUM_GPUS=1

# Paper YAML is already loaded into KBEVO_PAPER__* env in the preamble above,
# so the body defaults below just reference those env vars (with hardcoded
# fallbacks in case someone bypasses the preamble entirely).

# Batch / generation dimensions.
# Paper convention (Appendix B.1):
#   K = NUM_DB_ROLLOUTS       — phase 1 KB rollouts per question       (e.g. 4)
#   N = NUM_GENERATIONS       — phase 2 QA rollouts per question       (e.g. 32; must be divisible by K)
#   M = N / K                  — phase 2 QA rollouts per (question, KB) (e.g. 8)
#   Total generations per question = K + N                              (e.g. 4 + 32 = 36)
NUM_GENERATIONS=${KBEVO_PAPER__rollouts__N_phase2_qa_per_question:-32}
NUM_DB_ROLLOUTS=${KBEVO_PAPER__rollouts__K_phase1_kb_per_question:-4}
TOTAL_BATCH_SIZE=${KBEVO_PAPER__batching__effective_batch_size:-512}

PER_DEVICE_TRAIN_BATCH_SIZE=${KBEVO_PAPER__batching__per_device_train_batch_size:-16}
PER_DEVICE_EVAL_BATCH_SIZE=32
VLLM_GPU_MEMORY_UTILIZATION=${KBEVO_PAPER__vllm__vllm_gpu_memory_utilization:-0.6}

# Training hyperparameters (paper recipe; do NOT vary with GPU type).
LOSS_TYPE="grpo"
BETA=${KBEVO_PAPER__optimizer__beta:-0.0}
LEARNING_RATE=${KBEVO_PAPER__optimizer__learning_rate:-5e-6}
MAX_STEPS=${KBEVO_PAPER__epochs_or_steps__max_steps:-500}
NUM_TRAIN_EPOCHS=${KBEVO_PAPER__epochs_or_steps__num_train_epochs:-100}
TRAIN_SIZE=${KBEVO_PAPER__data__train_examples:-7000}
EVAL_SIZE=${KBEVO_PAPER__data__eval_examples:-100}
MAX_COMPLETION_LENGTH=${KBEVO_PAPER__rollouts__max_completion_length:-1024}

# Sampling.
TOP_P=${KBEVO_PAPER__sampling__top_p:-0.95}
TEMPERATURE=${KBEVO_PAPER__sampling__temperature:-1}
SAMPLING_TOP_K=${KBEVO_PAPER__sampling__sampling_top_k:-4}

# ── Logging / checkpointing ───────────────────────────────────────────────────
LOGGING_STEPS=5
SAVE_STEPS=25
EVAL_STEPS=100

# ── Core LMLM flags ───────────────────────────────────────────────────────────
TOOLS="--tools"               # tools enabled by default; pass --no_tools to disable
TWO_PHASE=""
RETRIEVAL_THRESHOLD=0.6
RETRIEVAL_TOP_K=1
REWARD_FUNC="em"
PHASE1_REWARD_TYPE="binary"
PHASE1_PROMPT_TYPE="sft"
PHASE1_DB_WEIGHT_MODE="count"  # none | fixed | dynamic | count | count_dynamic
USE_CHAT_TEMPLATE=""

# ── Ablation flags (off by default) ───────────────────────────────────────────
USE_ADAPTIVE_K=False
USE_INVERSES=""
VANILLA_GRPO=""
RETURN_TRIPLES=""
TIER_PATH=""
TIER_MIN_SCORE=1
TIER_MAX_SCORE=7
CURRICULUM=""
CURRICULUM_PHASES="5-7,3-7,1-7"
CURRICULUM_STEPS="0.33,0.67"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu_type)              GPU_TYPE="$2";               shift 2 ;;
        --model_path)            MODEL_PATH="$2";             shift 2 ;;
        --database_path)         DATABASE_PATH="$2";          shift 2 ;;
        --save_dir)              SAVE_DIR="$2";               shift 2 ;;
        --num_gpus)              NUM_GPUS="$2"; _NUM_GPUS_USER_SET=1; shift 2 ;;
        --dataset_name)          DATASET_NAME="$2";           shift 2 ;;
        --num_train_epochs)      NUM_TRAIN_EPOCHS="$2";       shift 2 ;;
        --max_steps)             MAX_STEPS="$2";              shift 2 ;;
        --train_size)            TRAIN_SIZE="$2";             shift 2 ;;
        --total_batch_size)      TOTAL_BATCH_SIZE="$2";       shift 2 ;;
        --per_device_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; _PER_DEVICE_USER_SET=1; shift 2 ;;
        --vllm_gpu_memory_utilization)
                                 VLLM_GPU_MEMORY_UTILIZATION="$2"; _VLLM_MEM_USER_SET=1; shift 2 ;;
        --num_generations)       NUM_GENERATIONS="$2";        shift 2 ;;
        --num_db_rollouts)       NUM_DB_ROLLOUTS="$2";        shift 2 ;;
        --learning_rate)         LEARNING_RATE="$2";          shift 2 ;;
        --max_completion_length|--max-completion-length)
                                  MAX_COMPLETION_LENGTH="$2";  shift 2 ;;
        --retrieval_threshold|--retrieval-threshold)
                                  RETRIEVAL_THRESHOLD="$2";    shift 2 ;;
        --retrieval_top_k|--retrieval-top-k|--top-k)
                                  RETRIEVAL_TOP_K="$2";        shift 2 ;;
        --sampling_top_k|--sampling-top-k)
                                  SAMPLING_TOP_K="$2";         shift 2 ;;
        # Backward compatibility: this used to control generation sampling while
        # run names described it as retrieval top-k. Preserve the old sampling
        # effect and, for positive values, make the advertised retrieval setting
        # effective too. Values such as 0/-1 remain valid sampling-only controls.
        --top_k)                 SAMPLING_TOP_K="$2"
                                  if [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                                      RETRIEVAL_TOP_K="$2"
                                  else
                                      echo "Warning: legacy --top_k=$2 only sets sampling top-k; use --retrieval-top-k for DB retrieval." >&2
                                  fi
                                  shift 2 ;;
        --reward_func)           REWARD_FUNC="$2";            shift 2 ;;
        --phase1_reward_type)    PHASE1_REWARD_TYPE="$2";     shift 2 ;;
        --phase1_prompt_type)    PHASE1_PROMPT_TYPE="$2";     shift 2 ;;
        --phase1_db_weight_mode) PHASE1_DB_WEIGHT_MODE="$2";  shift 2 ;;
        # Core flags
        --two_phase)             TWO_PHASE="--two_phase";     shift 1 ;;
        --no_two_phase|--no-two-phase)
                                  TWO_PHASE="";                shift 1 ;;
        --use_chat_template)     USE_CHAT_TEMPLATE="--use_chat_template"; shift 1 ;;
        --no_tools)              TOOLS="";                    shift 1 ;;
        # Ablation flags
        --use_adaptive_k)
                                  if [[ $# -gt 1 && "$2" != --* ]]; then
                                      case "$2" in
                                          True|true|TRUE|1|yes|Yes|YES)
                                              USE_ADAPTIVE_K=True ;;
                                          False|false|FALSE|0|no|No|NO)
                                              USE_ADAPTIVE_K=False ;;
                                          *)
                                              echo "Invalid value for --use_adaptive_k: $2" >&2
                                              exit 1 ;;
                                      esac
                                      shift 2
                                  else
                                      USE_ADAPTIVE_K=True; shift 1
                                  fi ;;
        --adaptive_k|--adaptive-k)
                                  USE_ADAPTIVE_K=True;          shift 1 ;;
        --use_inverses|--use-inverses)
                                  USE_INVERSES="--use_inverses"; shift 1 ;;
        --no_use_inverses|--no-use-inverses|--no_inverses|--no-inverses)
                                  USE_INVERSES="";               shift 1 ;;
        --vanilla_grpo)          VANILLA_GRPO="--vanilla_grpo"; shift 1 ;;
        --return_triples)        RETURN_TRIPLES="--return_triples"; shift 1 ;;
        --tier_path)             TIER_PATH="$2";              shift 2 ;;
        --tier_min_score)        TIER_MIN_SCORE="$2";         shift 2 ;;
        --tier_max_score)        TIER_MAX_SCORE="$2";         shift 2 ;;
        --curriculum)            CURRICULUM="--curriculum";   shift 1 ;;
        --curriculum_phases)     CURRICULUM_PHASES="$2";      shift 2 ;;
        --curriculum_steps)      CURRICULUM_STEPS="$2";       shift 2 ;;
        # Misc
        --debug)                 DEBUG=1;                     shift 1 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── NUM_GPUS resolution ──────────────────────────────────────────────────────
# Precedence for NUM_GPUS:
#   1) --num_gpus <N> (explicit CLI)
#   2) $SLURM_GPUS_ON_NODE                 (set by SLURM inside a job)
#   3) count $CUDA_VISIBLE_DEVICES commas   (set by SLURM or user)
#   4) nvidia-smi count                    (local execution)
#   5) default 1
if [[ -z "${_NUM_GPUS_USER_SET:-}" && "${NUM_GPUS}" == "1" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        NUM_GPUS="${SLURM_GPUS_ON_NODE}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
    else
        _detected=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [[ ${_detected} -gt 0 ]]; then
            NUM_GPUS=${_detected}
        fi
    fi
fi
ACCEL_CONFIG="configs/accelerate/multi_gpu_${NUM_GPUS}.yaml"

# ── GPU-type presets (hardware defaults only — NEVER scientific hyperparameters) ─
# Order of precedence for `PER_DEVICE_TRAIN_BATCH_SIZE` and
# `VLLM_GPU_MEMORY_UTILIZATION`:
#   paper/body default → GPU-type preset → explicit CLI --flag (wins).
# The `_*_USER_SET` markers ensure a user-supplied CLI value is NOT clobbered
# by the preset that runs after the parser.
case "$GPU_TYPE" in
    B200)
        if [[ "${MODEL_PATH}" == *"1.7B"* ]]; then
            [[ -z "${_PER_DEVICE_USER_SET:-}" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=16
            [[ -z "${_VLLM_MEM_USER_SET:-}"   ]] && VLLM_GPU_MEMORY_UTILIZATION=0.4
        elif [[ "${MODEL_PATH}" == *"4B"* ]]; then
            [[ -z "${_PER_DEVICE_USER_SET:-}" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=8
            [[ -z "${_VLLM_MEM_USER_SET:-}"   ]] && VLLM_GPU_MEMORY_UTILIZATION=0.15
        elif [[ "${MODEL_PATH}" == *"8B"* ]]; then
            [[ -z "${_PER_DEVICE_USER_SET:-}" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=4
            [[ -z "${_VLLM_MEM_USER_SET:-}"   ]] && VLLM_GPU_MEMORY_UTILIZATION=0.2
        elif [[ "${MODEL_PATH}" == *"382M"* ]]; then
            [[ -z "${_PER_DEVICE_USER_SET:-}" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=256
            [[ -z "${_VLLM_MEM_USER_SET:-}"   ]] && VLLM_GPU_MEMORY_UTILIZATION=0.15
        else
            echo "Warning: unrecognized model size for ${GPU_TYPE} preset: ${MODEL_PATH}. Falling back to defaults; pass --per_device_batch_size / --vllm_gpu_memory_utilization to tune." >&2
        fi ;;
    H100)
        [[ -z "${_PER_DEVICE_USER_SET:-}" ]] && PER_DEVICE_TRAIN_BATCH_SIZE=8
        [[ -z "${_VLLM_MEM_USER_SET:-}"   ]] && VLLM_GPU_MEMORY_UTILIZATION=0.15 ;;
    "")
        ;;
    *)
        echo "Warning: unknown --gpu_type=${GPU_TYPE}; using paper defaults." >&2 ;;
esac

if [ -n "${DEBUG}" ]; then
    echo "Debug mode: real smoke run (paper rollout preserved, tiny optimizer batch)"
    # Real smoke:
    #   * Preserve the paper rollout structure (N=32, K=4, M=8) so the full
    #     two-phase code path is exercised.
    #   * Shrink the OPTIMIZER batch to 32 (down from paper 512) so one step
    #     costs 32 questions × 36 gens instead of 512 × 36.
    #   * 1 optimizer step; log/save/eval each step; isolated per-invocation
    #     output dir so a debug run never resumes from an older debug ckpt.
    MAX_STEPS=1
    TOTAL_BATCH_SIZE=$(( NUM_GENERATIONS * NUM_GPUS ))
    TRAIN_SIZE=$TOTAL_BATCH_SIZE
    EVAL_SIZE=1
    LOGGING_STEPS=1
    SAVE_STEPS=1
    EVAL_STEPS=1
    if [[ -z "${_PER_DEVICE_USER_SET:-}" ]]; then
        PER_DEVICE_TRAIN_BATCH_SIZE=$NUM_GENERATIONS
        # Guard against pathological NUM_GENERATIONS: floor at 1.
        [[ $PER_DEVICE_TRAIN_BATCH_SIZE -lt 1 ]] && PER_DEVICE_TRAIN_BATCH_SIZE=1
    fi
    DEBUG_TAG="debug_${SLURM_JOB_ID:-local}_$(date +%s)"
fi

# Only synthesize CUDA_VISIBLE_DEVICES for local runs. If SLURM (or the user)
# has already set it, we honor that allocation exactly — never overwrite.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Derive GRADIENT_ACCUMULATION_STEPS from TOTAL_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS)))
if [ "$((GRADIENT_ACCUMULATION_STEPS * PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS))" -ne "${TOTAL_BATCH_SIZE}" ]; then
    echo "Error: TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} is not divisible by PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE} * NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
fi
echo "  GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS} (= ${TOTAL_BATCH_SIZE} / (${PER_DEVICE_TRAIN_BATCH_SIZE} * ${NUM_GPUS}))"

# ── Output directory ──────────────────────────────────────────────────────────
B=$((TOTAL_BATCH_SIZE / NUM_GENERATIONS))
M=$((NUM_GENERATIONS / NUM_DB_ROLLOUTS))

# Core name: if the last path component looks like a checkpoint, prepend the parent dir (actual model name)
MODEL_BASENAME="${MODEL_PATH##*/}"
if [[ "${MODEL_BASENAME}" == checkpoint-* ]]; then
    MODEL_BASENAME="$(basename "$(dirname "${MODEL_PATH}")")-${MODEL_BASENAME}"
fi
OUTPUT_DIR="${SAVE_DIR}/${MODEL_BASENAME}-${LOSS_TYPE}-tbs${TOTAL_BATCH_SIZE}-N${NUM_GENERATIONS}-K${NUM_DB_ROLLOUTS}-B${B}-M${M}-b${BETA}-lr${LEARNING_RATE}-step${MAX_STEPS}-n${TRAIN_SIZE}-${REWARD_FUNC}"
if [ -n "${TWO_PHASE}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-2ph"
    [ "${PHASE1_REWARD_TYPE}" != "binary" ] && OUTPUT_DIR="${OUTPUT_DIR}-rw${PHASE1_REWARD_TYPE}"
    OUTPUT_DIR="${OUTPUT_DIR}-pr${PHASE1_PROMPT_TYPE}-w${PHASE1_DB_WEIGHT_MODE}"
fi
# Keep all three values explicit. Besides making runs self-describing, the new
# naming avoids auto-resuming legacy "topk" checkpoints whose name described
# retrieval k even though that value was only applied to generation sampling.
OUTPUT_DIR="${OUTPUT_DIR}-rth${RETRIEVAL_THRESHOLD}-rk${RETRIEVAL_TOP_K}-sk${SAMPLING_TOP_K}"

# Ablation suffix
[ "${USE_ADAPTIVE_K}" != "True" ] && OUTPUT_DIR="${OUTPUT_DIR}-nak"
[ -n "${TIER_PATH}" ]             && OUTPUT_DIR="${OUTPUT_DIR}-tier${TIER_MIN_SCORE}_${TIER_MAX_SCORE}"
[ -n "${CURRICULUM}" ]            && OUTPUT_DIR="${OUTPUT_DIR}-curric"
[ -n "${USE_INVERSES}" ]          && OUTPUT_DIR="${OUTPUT_DIR}-inv"
[ -n "${VANILLA_GRPO}" ]          && OUTPUT_DIR="${OUTPUT_DIR}-vanilla"
# Debug runs land under an isolated per-job subdir so LAST_CKPT below can never
# resurrect an earlier debug checkpoint.
[ -n "${DEBUG}" ]                 && OUTPUT_DIR="${OUTPUT_DIR}-debug/${DEBUG_TAG}"

# ── Flag resolution ───────────────────────────────────────────────────────────
[ "${USE_ADAPTIVE_K}" = "True" ] && ADAPTIVE_K="--adaptive_k" || ADAPTIVE_K=""
[ -n "${USE_INVERSES}" ] && INVERSES_STATUS="on" || INVERSES_STATUS="off"

# ── Resume from checkpoint ────────────────────────────────────────────────────
LAST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
RESUME_FROM_CHECKPOINT=${LAST_CKPT:+"--resume_from_checkpoint=${LAST_CKPT}"}

# ── Resolved configuration ────────────────────────────────────────────────────
echo "════════════════ Resolved GRPO configuration ════════════════"
echo "  Model init      : ${MODEL_PATH}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Debug mode      : $([[ -n "$DEBUG" ]] && echo 'ON (real smoke: max_steps=1)' || echo 'off')"
echo "  Resume from     : ${RESUME_FROM_CHECKPOINT:-none}"
echo "  ── Hardware ──"
echo "  GPUs / preset   : ${NUM_GPUS} × ${GPU_TYPE:-default}"
echo "  Per-device bsz  : ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  Grad accum      : ${GRADIENT_ACCUMULATION_STEPS}"
echo "  vLLM mem util   : ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "  ── Paper recipe (Appendix B.1) ──"
echo "  Total batch     : ${TOTAL_BATCH_SIZE}    (optimizer effective batch)"
echo "  N (QA rollouts) : ${NUM_GENERATIONS}"
echo "  K (KB rollouts) : ${NUM_DB_ROLLOUTS}"
echo "  M (QA per KB)   : ${M}"
echo "  Learning rate   : ${LEARNING_RATE}"
echo "  Max steps       : ${MAX_STEPS}"
echo "  Beta            : ${BETA}"
echo "  Reward          : ${REWARD_FUNC}"
echo "  Two-phase       : ${TWO_PHASE:-off}"
echo "  Retrieval       : top-k=${RETRIEVAL_TOP_K}, threshold=${RETRIEVAL_THRESHOLD}, adaptive=${USE_ADAPTIVE_K}"
echo "  Sampling top-k  : ${SAMPLING_TOP_K}"
echo "  Inverses        : ${INVERSES_STATUS}"
echo "  Phase-1 prompt  : ${PHASE1_PROMPT_TYPE}"
echo "═════════════════════════════════════════════════════════════"

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
  --num_processes=${NUM_GPUS} \
  --config_file=${ACCEL_CONFIG} \
  src/grpo_train.py \
  --model_path="${MODEL_PATH}" \
  --dataset_name="${DATASET_NAME}" \
  --database_path="${DATABASE_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_generations=${NUM_GENERATIONS} \
  --num_generations_eval=${NUM_GENERATIONS} \
  --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --max_completion_length=${MAX_COMPLETION_LENGTH} \
  --logging_steps=${LOGGING_STEPS} \
  --vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
  --use_vllm \
  --vllm_mode=colocate \
  --gradient_checkpointing \
  --do_eval \
  --log_completions \
  --beta=${BETA} \
  --learning_rate=${LEARNING_RATE} \
  --loss_type=${LOSS_TYPE} \
  --max_grad_norm=1.0 \
  --warmup_ratio=0.1 \
  --lr_scheduler_type=cosine \
  --vllm_max_model_length=4096 \
  --train_size=${TRAIN_SIZE} \
  --eval_size=${EVAL_SIZE} \
  --top_p=${TOP_P} \
  --temperature=${TEMPERATURE} \
  --top_k=${SAMPLING_TOP_K} \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --max_steps=${MAX_STEPS} \
  --save_strategy=steps \
  --save_total_limit=5 \
  --save_steps=${SAVE_STEPS} \
  --eval_strategy=steps \
  --eval_steps=${EVAL_STEPS} \
  --retrieval_threshold=${RETRIEVAL_THRESHOLD} \
  --retrieval_top_k=${RETRIEVAL_TOP_K} \
  --reward_func=${REWARD_FUNC} \
  --phase1_reward_type=${PHASE1_REWARD_TYPE} \
  --phase1_prompt_type=${PHASE1_PROMPT_TYPE} \
  --num_db_rollouts=${NUM_DB_ROLLOUTS} \
  --phase1_db_weight_mode=${PHASE1_DB_WEIGHT_MODE} \
  --tier_min_score=${TIER_MIN_SCORE} \
  --tier_max_score=${TIER_MAX_SCORE} \
  ${TWO_PHASE} \
  ${TOOLS} \
  ${USE_CHAT_TEMPLATE} \
  ${ADAPTIVE_K} \
  ${VANILLA_GRPO} \
  ${RETURN_TRIPLES} \
  ${USE_INVERSES} \
  ${CURRICULUM} \
  ${RESUME_FROM_CHECKPOINT} \
  ${TIER_PATH:+--tier_path=${TIER_PATH}} \
  $([ -n "${CURRICULUM}" ] && echo "--curriculum_phases=${CURRICULUM_PHASES} --curriculum_steps=${CURRICULUM_STEPS}")

echo "Training completed!"
