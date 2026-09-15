# KBevo GRPO variants

This directory holds portable SLURM wrappers for the GRPO runs and ablations
that appear (or informed the design decisions) in the KBevo paper
[arXiv:2608.26386](https://arxiv.org/pdf/2608.26386) (COLM 2026). Every
wrapper is a thin front-end that forwards flags to the canonical training
script `scripts/train_grpo.sh`; the GRPO training body itself is never
copied.

## Canonical rollout / batch convention (paper Appendix B.1)

| Symbol | Meaning                                                | Wrapper flag           | Canonical value |
|--------|--------------------------------------------------------|------------------------|-----------------|
| `N`    | Phase-2 QA rollouts per question                       | `--num_generations`    | 32              |
| `K`    | Phase-1 KB rollouts per question                       | `--num_db_rollouts`    | 4               |
| `M`    | Phase-2 QA rollouts per (question, KB) — `M = N / K`   | derived                | 8               |
| —      | Total generations per question = `K + N`               | derived                | 36              |
| —      | Optimizer effective batch (examples per gradient step) | `--total_batch_size`   | 512             |

The rollout count (36 per question) and the optimizer effective batch (512
examples per gradient step) are **independent knobs** — the paper's
`TOTAL_BATCH_SIZE=512` refers only to the optimizer / gradient-step batch,
not the rollout structure. New runs use `M8` in their output-directory
names; existing historical checkpoint directories are left as-is.

## Variants

| Variant                    | Purpose                                                                    | Model size | Paper role                    | Required inputs (in addition to `KBEVO_ENV`, `KBEVO_CKPT_ROOT`) | Launch command                                                                        |
|----------------------------|----------------------------------------------------------------------------|------------|-------------------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `train_two_phase_1.7b.slurm` | Paper main two-phase GRPO run on Qwen3-1.7B.                             | 1.7B       | Main paper checkpoint         | `KBEVO_SFT_1_7B_CKPT`                                           | `sbatch --partition=<p> --gres=gpu:4 scripts/grpo_variants/train_two_phase_1.7b.slurm` |
| `train_two_phase_4b.slurm`   | Paper main two-phase GRPO run on Qwen3-4B (SFT step 735 init).           | 4B         | Main paper checkpoint         | `KBEVO_SFT_4B_CKPT`                                             | `sbatch --partition=<p> --gres=gpu:4 scripts/grpo_variants/train_two_phase_4b.slurm`   |
| `train_one_phase.slurm`      | Ablation: 1-phase SFT init + static pre-built KB, no phase-1 rollouts.   | 1.7B       | Ablation (not released)       | `KBEVO_ONE_PHASE_SFT_1_7B_CKPT`, `KBEVO_ONE_PHASE_DB` *(private)* | `sbatch --partition=<p> --gres=gpu:4 scripts/grpo_variants/train_one_phase.slurm`      |
| `train_vanilla_grpo.slurm`   | Ablation: vanilla-GRPO advantage/update, N=16, no inverses.              | 1.7B       | Ablation (not released)       | `KBEVO_SFT_1_7B_CKPT`                                           | `sbatch --partition=<p> --gres=gpu:1 scripts/grpo_variants/train_vanilla_grpo.slurm`   |
| `train_zero_rl.slurm`        | Ablation: GRPO from raw Qwen3-1.7B base (no SFT) with a chat-template prompt. | 1.7B  | Ablation (not released)       | none (uses public `Qwen/Qwen3-1.7B`)                            | `sbatch --partition=<p> --gres=gpu:1 scripts/grpo_variants/train_zero_rl.slurm`        |
| `train_curriculum.slurm`     | Ablation: tier-filtered curriculum on the full HotpotQA train set (90k). | 1.7B       | Ablation (not released)       | `KBEVO_SFT_1_7B_CKPT`, `KBEVO_TIER_PATH` *(private)*            | `sbatch --partition=<p> --gres=gpu:1 scripts/grpo_variants/train_curriculum.slurm`     |

Legend: *private* = the artifact is not part of the paper's public release
and this repository does not ship a download recipe for it.

## Configuration

All wrappers read paths from `configs/cluster.env` (copied from
`configs/cluster.env.example`) — nothing is hard-coded to a site or user.
The key variables are:

- **Core** — `KBEVO_ENV`, `KBEVO_CKPT_ROOT`, `KBEVO_SFT_1_7B_CKPT`,
  `KBEVO_SFT_4B_CKPT`.
- **Ablation-only** — `KBEVO_ONE_PHASE_SFT_1_7B_CKPT`, `KBEVO_ONE_PHASE_DB`,
  `KBEVO_TIER_PATH`.
- **W&B** — `WANDB_ENTITY`, `WANDB_PROJECT` (both unset ⇒ W&B disabled).

Partition, account, QoS, GPU type, and Slurm constraint are supplied
through the `sbatch` submission command, not with `#SBATCH` directives,
because `#SBATCH` lines are parsed before shell variables are expanded.

## Outputs

Every variant writes its GRPO checkpoint tree under
`$KBEVO_CKPT_ROOT/grpo_1.7b/…`, `.../grpo_4b/…`, or, for the ablations that
set `MODEL_PATH` via env and skip `--model_size`, `.../grpo_custom/…`. The
output-directory name is
composed from the actual (post-parse) hyperparameters, so ablations and the
main run land in visibly distinct subtrees. Concretely the canonical run
produces a directory whose name includes `…-N32-K4-B16-M8-…-rk4-sk4-…-inv`.

## Local debugging

To dry-run without submitting to SLURM, add `--debug` — the wrapper
forwards it and the trainer switches to `TRAIN_SIZE=1000`, `EVAL_SIZE=10`,
which still exercises the same rollout / advantage / update path:

```bash
bash scripts/grpo_variants/train_two_phase_1.7b.slurm --debug
```
