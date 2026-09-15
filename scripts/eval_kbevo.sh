#!/bin/bash
# KBevo evaluation — single self-contained script for all four datasets.
# Runs the paper-canonical two_phase eval on HotpotQA, MuSiQue, 2WikiMultiHopQA,
# and PopQA long-tail.
#
# Foreground:  bash scripts/eval_kbevo.sh --model_path /path/to/kbevo/ckpt
# SLURM:       sbatch --partition=<your_partition> --gres=gpu:1 scripts/eval_kbevo.sh --model_path <ckpt>
#
# Args:
#   --model_path   <path>                          KBevo checkpoint dir (required)
#   --datasets     hotpotqa,musique,2wiki,popqa    subset (default: all four)
#   --num_samples  <int>                           samples per dataset (default 1000)
#   --save_version <tag>                           output-file suffix (default _eval_kbevo)
#   --output-dir   <path>                          preds root (default ./output/main_tables)
#
# Env (see configs/cluster.env.example):
#   KBEVO_ENV=kbevo               conda env
#   KBEVO_2WIKI_DB=<path>         pre-built 2wiki phase-1 DB (only for --method lmlm on 2wiki)
#   POPQA_CORPUS_PATH=<path>      local Wikipedia contexts jsonl (else auto-downloads from HF)

#SBATCH -J kbevo_eval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --requeue

set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/eval_kbevo.sh" ]]; then
    KBEVO_ROOT="$SLURM_SUBMIT_DIR"
else
    KBEVO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
fi
cd "$KBEVO_ROOT"
[[ -f configs/cluster.env ]] && source configs/cluster.env

# Load paper YAML defaults (single source of truth). Any CLI flag below wins.
_PAPER_YAML="$KBEVO_ROOT/configs/paper/eval_kbevo.yaml"
if [[ -f "$_PAPER_YAML" ]]; then
    eval "$(python "$KBEVO_ROOT/scripts/_paper_config.py" "$_PAPER_YAML")"
fi

# --- Defaults (YAML if loaded, else hardcoded fallback) ---
DATASETS="hotpotqa,musique,2wiki,popqa"
NUM_SAMPLES=1000
MODEL_PATH=""
SAVE_VERSION="_eval_kbevo"
OUTPUT_DIR="${OUTPUT_DIR:-./output/main_tables}"
SEED=${KBEVO_PAPER__seed:-42}
TOP_K=${KBEVO_PAPER__retrieval__retrieval_top_k:-4}
SIMILARITY_THRESHOLD=${KBEVO_PAPER__retrieval__similarity_threshold:-0.6}
MAX_TOKENS=${KBEVO_PAPER__generation__max_tokens:-1024}
BATCH_SIZE=${KBEVO_PAPER__batching__batch_size:-64}
SAVE_EVERY=${KBEVO_PAPER__batching__save_every:-64}

# --- CLI overrides (CLI beats YAML) ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)   MODEL_PATH="$2";   shift 2 ;;
        --datasets)     DATASETS="$2";     shift 2 ;;
        --num_samples)  NUM_SAMPLES="$2";  shift 2 ;;
        --save_version) SAVE_VERSION="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --seed)                                 SEED="$2";                 shift 2 ;;
        --top-k|--top_k)                        TOP_K="$2";                shift 2 ;;
        --similarity-threshold|--similarity_threshold)
                                                SIMILARITY_THRESHOLD="$2"; shift 2 ;;
        --max-tokens|--max_tokens)              MAX_TOKENS="$2";           shift 2 ;;
        --batch-size|--batch_size)              BATCH_SIZE="$2";           shift 2 ;;
        --save-every|--save_every)              SAVE_EVERY="$2";           shift 2 ;;
        -h|--help)      sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model_path is required (bash $0 --help for usage)" >&2; exit 2
fi
# Accept either a local checkpoint directory OR a Hugging Face repo id
# (e.g. "kilian-group/KBevo-Qwen3-1.7B-GRPO"). For an HF id, snapshot_download
# the full snapshot into the HF cache and rewrite MODEL_PATH to the local path.
if [[ ! -d "$MODEL_PATH" ]]; then
    if [[ "$MODEL_PATH" == */* && "$MODEL_PATH" != /* && "$MODEL_PATH" != ./* ]]; then
        echo "  Resolving Hugging Face repo id: $MODEL_PATH"
        LOCAL_SNAPSHOT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_PATH'))" 2>&1) || {
            echo "ERROR: could not snapshot_download HF repo '$MODEL_PATH'" >&2
            echo "$LOCAL_SNAPSHOT" >&2
            exit 2
        }
        echo "  → local snapshot: $LOCAL_SNAPSHOT"
        MODEL_PATH="$LOCAL_SNAPSHOT"
    else
        echo "ERROR: --model_path '$MODEL_PATH' is neither a local directory nor a valid HF repo id (owner/name)" >&2
        exit 2
    fi
fi

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

echo "=================================================="
echo " KBevo evaluation"
echo "   model      = $MODEL_PATH"
echo "   datasets   = $DATASETS"
echo "   samples    = $NUM_SAMPLES  per dataset"
echo "   output     = $OUTPUT_DIR"
echo "   seed       = $SEED"
echo "   top_k      = $TOP_K"
echo "   threshold  = $SIMILARITY_THRESHOLD"
echo "   max_tokens = $MAX_TOKENS"
echo "   batch_size = $BATCH_SIZE"
echo "=================================================="

IFS=',' read -ra DSLIST <<< "$DATASETS"
for DS_RAW in "${DSLIST[@]}"; do
    DS="${DS_RAW// /}"
    # Per-dataset config (split, setting, dataset-source).
    SETTING="distractor"
    DATASET_SRC_FLAG=""
    case "$DS" in
        hotpotqa)  SPLIT="dev" ;;
        musique)   SPLIT="dev" ;;
        2wiki)     SPLIT="dev" ;;
        popqa)     SPLIT="test" ; SETTING="long_tail" ; DATASET_SRC_FLAG="--dataset-source hf" ;;
        *)         echo "ERROR: unknown dataset '$DS' (expected hotpotqa|musique|2wiki|popqa)" >&2 ; exit 2 ;;
    esac

    echo ""
    echo "----- eval on $DS (split=$SPLIT, setting=$SETTING, num_samples=$NUM_SAMPLES) -----"
    python src/eval_multihop.py \
        $DATASET_SRC_FLAG \
        --model-path "$MODEL_PATH" \
        --method two_phase \
        --dataset "$DS" \
        --split "$SPLIT" \
        --setting "$SETTING" \
        --max-tokens $MAX_TOKENS \
        --batch-size $BATCH_SIZE \
        --total-count $NUM_SAMPLES \
        --output-dir "${OUTPUT_DIR}/" \
        --save-version "$SAVE_VERSION" \
        --seed $SEED \
        --save-every $SAVE_EVERY \
        --start-index 0 \
        --top-k $TOP_K \
        --similarity-threshold $SIMILARITY_THRESHOLD \
        --use-inverses \
        --use-train-params \
        --concat-all-db \
        --use-contexts all \
        --eval \
        --resume
done

echo ""
echo "=== Done. Results under: $OUTPUT_DIR/two_phase/{hotpotqa,musique,2wiki,popqa}/<model-name>/ ==="
