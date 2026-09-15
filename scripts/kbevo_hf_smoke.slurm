#!/bin/bash
# KBevo Hugging Face smoke test — loads one released KBevo checkpoint from
# the Hub, verifies BF16 stored weights, then runs a tiny 4-dataset two-phase
# eval (hotpotqa, musique, 2wiki, popqa) via scripts/eval_kbevo.sh.
#
# Scheduler-agnostic. Run directly with bash:
#
#   bash scripts/kbevo_hf_smoke.slurm \
#       --hf_repo kilian-group/KBevo-Qwen3-1.7B-GRPO \
#       --num_samples 2
#
# Or through the optional SLURM wrapper:
#
#   bash scripts/submit_slurm.sh hf-smoke -- \
#       --hf_repo kilian-group/KBevo-Qwen3-1.7B-GRPO \
#       --num_samples 2
#
# CLI args:
#   --hf_repo <owner/name>   (required)  e.g. kilian-group/KBevo-Qwen3-1.7B-GRPO
#   --num_samples <int>      (default 5) examples per dataset
#   --output_dir <path>      (default <repo>/output/hf_smoke)
#
# The legacy env-var interface (`HF_REPO=...`, `NUM_SAMPLES=...`, `OUTPUT_DIR=...`)
# is still honoured as a fallback for backward compatibility, but --flags win.
#
# Optional env:
#   KBEVO_ENV=<conda-env>    conda env to activate (default: kbevo)

#SBATCH -J kbevo_hf_smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --requeue

set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/kbevo_hf_smoke.slurm" ]]; then
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

# ── Parse CLI flags (env vars are the fallback) ──────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_repo)      HF_REPO="$2";      shift 2 ;;
        --num_samples)  NUM_SAMPLES="$2";  shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        -h|--help)      sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
: "${HF_REPO:?Pass --hf_repo owner/name (or set HF_REPO env). Example: kilian-group/KBevo-Qwen3-1.7B-GRPO}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$KBEVO_ROOT/output/hf_smoke}"

# Job-scoped caches. Only these directories are removed on exit — never a
# path that could be shared or user-owned.
JOB_ID="${SLURM_JOB_ID:-local_$$}"
export VLLM_CACHE_ROOT="$HOME/.cache/kbevo/vllm_${JOB_ID}"
export TRITON_CACHE_DIR="$HOME/.cache/kbevo/triton_${JOB_ID}"
export TMPDIR="$HOME/.cache/kbevo/tmp_${JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR"
trap 'rm -rf "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR"' EXIT

echo "==================================================="
echo "  KBevo HF-load smoke test"
echo "  repo:         $HF_REPO"
echo "  samples/ds:   $NUM_SAMPLES"
echo "  output:       $OUTPUT_DIR"
echo "  node:         $(hostname)"
echo "==================================================="

# Materialize the HF repo to a local snapshot dir (vLLM/verl prefer local).
# Only the last stdout line is captured; hub warnings do not leak into the
# path. The repo id passes through the environment, never Python source.
MODEL_PATH="$(HF_REPO="$HF_REPO" python - <<'PY' 2>/dev/null | tail -n 1
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ["HF_REPO"]))
PY
)"
if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: snapshot_download('$HF_REPO') did not resolve to a local directory." >&2
    exit 2
fi
echo "  snapshot path: $MODEL_PATH"

# Verify BF16 in the actually-stored safetensors weights + read generation config.
#
# We read the safetensors header directly, so we assert the DISK dtype of every
# tensor. Loading with `dtype=torch.bfloat16` would silently up/downcast an
# FP32/FP16 checkpoint to BF16 in RAM and mask a mis-uploaded weight file.
MODEL_PATH="$MODEL_PATH" python <<'PY'
import glob, json, os, struct
from transformers import GenerationConfig

mp = os.environ["MODEL_PATH"]

def read_safetensors_dtypes(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode())
    return {k: v["dtype"] for k, v in header.items() if k != "__metadata__"}

shards = sorted(glob.glob(os.path.join(mp, "*.safetensors")))
if not shards:
    raise SystemExit(f"no *.safetensors in {mp}")

all_dtypes: dict[str, int] = {}
n_tensors = 0
for shard in shards:
    for name, dt in read_safetensors_dtypes(shard).items():
        all_dtypes[dt] = all_dtypes.get(dt, 0) + 1
        n_tensors += 1

non_bf16 = {dt: c for dt, c in all_dtypes.items() if dt not in ("BF16", "bfloat16")}
if non_bf16:
    raise SystemExit(f"non-BF16 tensors on disk: {non_bf16} (out of {n_tensors})")
print(f"  ✓ all {n_tensors} tensors across {len(shards)} shard(s) are BF16 on disk")

gc = GenerationConfig.from_pretrained(mp)
print(f"  ✓ gen_config: do_sample={gc.do_sample} temp={gc.temperature} top_p={gc.top_p} top_k={gc.top_k}")
PY

# 4-dataset KBevo two-phase smoke, NUM_SAMPLES examples each.
bash "$KBEVO_ROOT/scripts/eval_kbevo.sh" \
    --model_path "$MODEL_PATH" \
    --datasets hotpotqa,musique,2wiki,popqa \
    --num_samples "$NUM_SAMPLES" \
    --save_version "_hf_smoke_$(basename "$HF_REPO")" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "==================================================="
echo "  DONE — check $OUTPUT_DIR/two_phase/*/*/results_*"
echo "==================================================="
