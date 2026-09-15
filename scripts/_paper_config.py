#!/usr/bin/env python3
"""Emit shell `export KEY=value` lines from one of configs/paper/*.yaml.

The launcher scripts (`scripts/train_sft.sh`, `scripts/train_grpo.sh`) invoke:

    eval "$(python scripts/_paper_config.py configs/paper/grpo_qwen3_1.7b.yaml)"

which sets `KBEVO_PAPER__<flat_dotted_key>` shell variables. The launcher
then maps each variable to its wrapper-local name AS A DEFAULT ONLY — an
explicit CLI flag ALWAYS wins.

This makes `configs/paper/*.yaml` the single authoritative source of the
scientific hyperparameters (LR, batch, N/K, retrieval, etc.). Editing the
YAML is enough; the wrapper picks up the change on next launch.

Fields we do NOT export (site-specific / runtime metadata):
    recipe_name, paper, paper_section, released_checkpoint, runtime,
    logging_and_checkpointing (unless the launcher opts in explicitly),
    initialization_step_note, effective_batch_size_note, ...
"""

from __future__ import annotations
import argparse
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("ERROR: pyyaml is required. `pip install pyyaml` or update environment.yml.\n")
    sys.exit(3)


# Fields we export from the YAML → shell env. Prefix keeps the paper values
# out of the launcher's normal env namespace.
PREFIX = "KBEVO_PAPER__"

# Some YAML fields are documentation, not hyperparameters. Skip them so we
# don't leak note-strings into the shell.
SKIP_TOP_KEYS = {"recipe_name", "paper", "paper_section", "released_checkpoint", "runtime"}
SKIP_NESTED_SUBSTRINGS = ("_note",)


def _flatten(prefix: str, node) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if any(s in k for s in SKIP_NESTED_SUBSTRINGS):
                continue
            out.extend(_flatten(f"{prefix}__{k}" if prefix else str(k), v))
    else:
        out.append((prefix, node))
    return out


def _fmt_value(v) -> str:
    """Render a Python value as a shell-safe token."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v).replace("+", "")
    return shlex.quote(str(v))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yaml_path", type=Path, help="Path to a configs/paper/*.yaml file")
    args = p.parse_args(argv)

    if not args.yaml_path.is_file():
        sys.stderr.write(f"ERROR: {args.yaml_path} not found\n"); return 2

    cfg = yaml.safe_load(args.yaml_path.read_text())
    if not isinstance(cfg, dict):
        sys.stderr.write(f"ERROR: {args.yaml_path} does not contain a top-level mapping\n"); return 2

    # Emit shell assignments. Order is stable (matches YAML iteration).
    for top_k, top_v in cfg.items():
        if top_k in SKIP_TOP_KEYS:
            continue
        for flat_k, v in _flatten(top_k, top_v):
            key = f"{PREFIX}{flat_k}"
            print(f"export {key}={_fmt_value(v)}")

    # Comment header for humans reading `bash -x` output.
    print(f'# ─ paper defaults loaded from {args.yaml_path} ─')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
