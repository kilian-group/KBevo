#!/bin/bash
# Optional SLURM submission convenience for KBevo.
#
# The canonical scripts (scripts/train_sft.sh, scripts/train_grpo.sh,
# scripts/eval_kbevo.sh, scripts/kbevo_hf_smoke.slurm) are scheduler-agnostic
# and can be run directly with `bash` on any machine. This wrapper is a THIN
# convenience layer that sources `configs/cluster.env` and translates the
# `SLURM_*` variables into `sbatch` command-line flags, so a user does not
# have to repeat --partition/--account/--qos/--gres/--constraint on every
# submission. `#SBATCH` directives inside the canonical scripts cannot
# expand shell variables — this wrapper is the supported way to make
# `cluster.env` take effect.
#
# Usage:
#   bash scripts/submit_slurm.sh <job> [args forwarded to the canonical script]
#
#   bash scripts/submit_slurm.sh sft-1.7b   --debug
#   bash scripts/submit_slurm.sh sft-4b     --debug
#   bash scripts/submit_slurm.sh grpo-1.7b  --debug
#   bash scripts/submit_slurm.sh grpo-4b    --debug
#   bash scripts/submit_slurm.sh eval       --model_path kilian-group/KBevo-Qwen3-1.7B-GRPO --num_samples 2
#   bash scripts/submit_slurm.sh hf-smoke   --hf_repo  kilian-group/KBevo-Qwen3-1.7B-GRPO --num_samples 2
#
# Everything after the <job> keyword is forwarded verbatim to the canonical
# script.
#
# Variables read from `configs/cluster.env` (all optional — omitted flags
# fall back to sbatch defaults):
#   SLURM_PARTITION       → sbatch --partition=<value>
#   SLURM_ACCOUNT         → sbatch --account=<value>
#   SLURM_QOS             → sbatch --qos=<value>
#   SLURM_CONSTRAINT      → sbatch --constraint=<value>
#   SLURM_TIME            → sbatch --time=<value>          (e.g. 03:00:00)
#   SLURM_SFT_GRES        → sbatch --gres=<value>  for sft-* jobs
#   SLURM_GRPO_GRES       → sbatch --gres=<value>  for grpo-* jobs
#   SLURM_EVAL_GRES       → sbatch --gres=<value>  for eval / hf-smoke
#
# Override precedence (highest first):
#   1) `--sbatch --flag=value` on this command line  (one-off overrides)
#   2) A variable already exported in the caller's shell
#   3) `configs/cluster.env`
#   4) omitted (sbatch default kicks in)
#
# The user's `configs/cluster.env` is git-ignored; only `cluster.env.example`
# is tracked.

set -eo pipefail
KBEVO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$KBEVO_ROOT"

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,44p' "$0"; exit 0
fi

# ── Save any SLURM_* vars pre-exported by the caller so cluster.env cannot
#    silently clobber a deliberate override.
declare -A _PRE
for _v in SLURM_PARTITION SLURM_ACCOUNT SLURM_QOS SLURM_CONSTRAINT SLURM_TIME \
          SLURM_SFT_GRES SLURM_GRPO_GRES SLURM_EVAL_GRES; do
    _PRE[$_v]="${!_v-}"
done

[[ -f configs/cluster.env ]] && source configs/cluster.env

# Restore any caller-provided override (env > cluster.env).
for _v in "${!_PRE[@]}"; do
    [[ -n "${_PRE[$_v]}" ]] && printf -v "$_v" '%s' "${_PRE[$_v]}"
done

JOB="$1"; shift

# Parse --sbatch KEY=VAL one-off overrides. Everything else forwards to the
# canonical script.
EXTRA_SBATCH=()
FORWARD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sbatch)
            if [[ $# -lt 2 ]]; then echo "ERROR: --sbatch needs an argument like --partition=my_p" >&2; exit 2; fi
            EXTRA_SBATCH+=("$2"); shift 2 ;;
        *)  FORWARD+=("$1"); shift ;;
    esac
done

case "$JOB" in
    sft-1.7b)   SCRIPT="scripts/train_sft.sh";                              ARGS=(--model_size 1.7B "${FORWARD[@]}"); GRES="${SLURM_SFT_GRES:-}"  ;;
    sft-4b)     SCRIPT="scripts/train_sft.sh";                              ARGS=(--model_size 4B   "${FORWARD[@]}"); GRES="${SLURM_SFT_GRES:-}"  ;;
    grpo-1.7b)  SCRIPT="scripts/grpo_variants/train_two_phase_1.7b.slurm";  ARGS=("${FORWARD[@]}");                    GRES="${SLURM_GRPO_GRES:-}" ;;
    grpo-4b)    SCRIPT="scripts/grpo_variants/train_two_phase_4b.slurm";    ARGS=("${FORWARD[@]}");                    GRES="${SLURM_GRPO_GRES:-}" ;;
    eval)       SCRIPT="scripts/eval_kbevo.sh";                             ARGS=("${FORWARD[@]}");                    GRES="${SLURM_EVAL_GRES:-}" ;;
    hf-smoke)   SCRIPT="scripts/kbevo_hf_smoke.slurm";                      ARGS=("${FORWARD[@]}");                    GRES="${SLURM_EVAL_GRES:-}" ;;
    *)
        echo "Unknown job: $JOB" >&2
        echo "Valid jobs: sft-1.7b, sft-4b, grpo-1.7b, grpo-4b, eval, hf-smoke" >&2
        exit 2 ;;
esac

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found — SLURM is not installed on this machine." >&2
    echo "  Run the canonical script directly instead:" >&2
    echo "    bash $SCRIPT ${ARGS[*]}" >&2
    exit 3
fi

# Compose sbatch flags. Only include a flag when its value is non-empty —
# empty --partition=/--account= would confuse sbatch on some sites. All
# generated flags go BEFORE the job-script path.
mkdir -p slurm
SBATCH_FLAGS=(--output="slurm/slurm-%j.out" --error="slurm/slurm-%j.err")
[[ -n "${SLURM_PARTITION:-}"  ]] && SBATCH_FLAGS+=(--partition="$SLURM_PARTITION")
[[ -n "${SLURM_ACCOUNT:-}"    ]] && SBATCH_FLAGS+=(--account="$SLURM_ACCOUNT")
[[ -n "${SLURM_QOS:-}"        ]] && SBATCH_FLAGS+=(--qos="$SLURM_QOS")
[[ -n "${SLURM_CONSTRAINT:-}" ]] && SBATCH_FLAGS+=(--constraint="$SLURM_CONSTRAINT")
[[ -n "${SLURM_TIME:-}"       ]] && SBATCH_FLAGS+=(--time="$SLURM_TIME")
[[ -n "${GRES:-}"             ]] && SBATCH_FLAGS+=(--gres="$GRES")
# One-off overrides come LAST so they take precedence when sbatch sees a
# repeated flag.
SBATCH_FLAGS+=("${EXTRA_SBATCH[@]}")

echo "══ Submitting to SLURM ══"
echo "  Job kind        : $JOB"
echo "  Canonical script: $SCRIPT"
echo "  Forwarded args  : ${ARGS[*]:-(none)}"
echo "  sbatch flags    : ${SBATCH_FLAGS[*]:-(none — sbatch defaults)}"
echo

exec sbatch "${SBATCH_FLAGS[@]}" "$SCRIPT" "${ARGS[@]}"
