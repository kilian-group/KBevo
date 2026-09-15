"""Run one KBevo two-phase inference over a small, self-contained example.

Given one question and a handful of passages, this script:

  1. Loads a released KBevo model (local checkpoint dir OR Hugging Face id).
  2. Calls the existing `TwoPhaseAgent` — the same code path used by
     `scripts/eval_kbevo.sh` — so we do NOT reimplement the two-phase
     algorithm.
  3. Writes a small JSON with:
       * the phase-1 extracted knowledge (KB triples),
       * the phase-2 tool trace (`<|db_*|>` lookup calls and returns),
       * the final short answer.

Usage
-----

    # Local checkpoint
    python examples/single_example.py \
        --model-path $HOME/kbevo_ckpts/grpo_1.7b/... \
        --output-path examples/my_output.json

    # Hugging Face repo id
    python examples/single_example.py \
        --model-path kilian-group/KBevo-Qwen3-1.7B-GRPO \
        --output-path examples/my_output.json

`--help` prints all options without requiring a GPU. The heavy imports
(torch, vLLM, transformers) are deferred until we actually need them, so the
help text alone works in a CPU-only environment.
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import sys
from pathlib import Path


# One canonical HotpotQA-style example that exercises Phase-1 KB extraction
# from short passages and a two-hop Phase-2 answer.
EXAMPLE_QUESTION = (
    "In addition to his puppeteering on The Dark Crystal, Dave Goelz is known "
    "for performing as which Muppet?"
)
EXAMPLE_CONTEXTS = [
    "Dave Goelz is an American puppeteer best known for his work with the "
    "Muppets. He performed The Great Gonzo, Bunsen Honeydew, Zoot, and Waldorf.",
    "The Dark Crystal is a 1982 American-British fantasy film directed by "
    "Jim Henson and Frank Oz. Dave Goelz was one of the film's puppeteers.",
    "The Great Gonzo, also known as Gonzo the Great, is a Muppet character "
    "who has been performed by Dave Goelz since 1976.",
    "Sesame Street is an American educational children's television series "
    "created by Joan Ganz Cooney and Lloyd Morrisett.",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one KBevo two-phase inference example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        default="kilian-group/KBevo-Qwen3-1.7B-GRPO",
        help="Local checkpoint directory OR Hugging Face repo id (owner/name).",
    )
    p.add_argument(
        "--output-path",
        default="examples/single_example_output.json",
        help="Where to write the resulting JSON.",
    )
    # NOTE: TwoPhaseAgent instantiates vLLM with a hard-coded seed=42; there
    # is no per-run seed hook. We do NOT expose --seed here to avoid an
    # advertised-but-ignored knob.
    p.add_argument(
        "--phase1-prompt-type",
        default="sft",
        help="Phase-1 prompt template key from data/prompts/database_creation.json.",
    )
    p.add_argument(
        "--similarity-threshold", type=float, default=0.6,
        help="Retrieval cosine-similarity threshold (paper: 0.6).",
    )
    p.add_argument("--top-k", type=int, default=4, help="Retrieval top-k (paper: 4).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument(
        "--use-inverses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include inverse triples in retrieval (paper: on).",
    )
    return p


def _resolve_model_path(model_path: str) -> str:
    """Return a local directory. If `model_path` is an HF id, materialize it."""
    p = Path(model_path)
    if p.is_dir():
        return str(p)
    if "/" in model_path and not model_path.startswith(("/", ".")):
        # Looks like `owner/name`; download it.
        from huggingface_hub import snapshot_download
        return snapshot_download(model_path)
    raise SystemExit(
        f"--model-path '{model_path}' is neither a local dir nor an owner/name HF repo id."
    )


def _agent_step_to_dict(step):
    """Convert one AgentStep dataclass to a JSON-serializable dict.

    Falls back to `dict(step)` / `vars(step)` if the object is not a
    dataclass — so this also copes with test mocks.
    """
    if dataclasses.is_dataclass(step) and not isinstance(step, type):
        return dataclasses.asdict(step)
    if isinstance(step, dict):
        return step
    try:
        return dict(vars(step))
    except TypeError:
        return {"repr": repr(step)}


def build_result(agent, answers, traces, model_path: str) -> dict:
    """Compose the JSON payload from the agent's post-run state.

    Isolated from `main()` so tests can drive it against a mocked agent.
    """
    trace0_steps = traces[0] if traces else []
    # `_phase1_info` / `_lookup_logs` are populated per-query by TwoPhaseAgent.
    # Fall back to empty lists if the agent skipped Phase-1 or did no lookups.
    phase1_info = getattr(agent, "_phase1_info", []) or []
    lookup_logs = getattr(agent, "_lookup_logs", []) or []
    phase1_triplets = phase1_info[0].get("triplets") if phase1_info else None
    phase2_lookups = lookup_logs[0] if lookup_logs else []

    return {
        "schema_version": 1,
        "question": EXAMPLE_QUESTION,
        "contexts": EXAMPLE_CONTEXTS,
        "answer": answers[0] if answers else None,
        "phase1_kb": phase1_triplets,
        "phase2_tool_trace": phase2_lookups,
        "raw_trace": [_agent_step_to_dict(s) for s in trace0_steps],
        "model_path": model_path,
    }


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _print_summary(payload: dict) -> None:
    """Human-readable summary of the demo output (matches what the README promises)."""
    print()
    print("── Question ──")
    print(f"  {payload['question']}")
    print()
    print("── Phase-1 KB triples ──")
    kb = payload.get("phase1_kb") or []
    if not kb:
        print("  (empty — model produced no phase-1 triples)")
    else:
        for t in kb[:20]:
            print(f"  {t}")
        if len(kb) > 20:
            print(f"  ... {len(kb) - 20} more triples in {payload.get('_json_path', 'the JSON output')}")
    print()
    print("── Phase-2 lookup trace ──")
    trace = payload.get("phase2_tool_trace") or []
    if not trace:
        print("  (empty — model answered without any phase-2 lookup)")
    else:
        for i, entry in enumerate(trace[:20], 1):
            print(f"  [{i}] {entry}")
        if len(trace) > 20:
            print(f"  ... {len(trace) - 20} more lookups in {payload.get('_json_path', 'the JSON output')}")
    print()
    print("── Final answer ──")
    print(f"  {payload.get('answer')!r}")


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Deferred heavy imports: `--help` must work without a GPU/torch/vLLM.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agent.two_phase_agent import TwoPhaseAgent  # noqa: E402

    resolved = _resolve_model_path(args.model_path)
    agent = TwoPhaseAgent(
        model_path=resolved,
        phase1_prompt_type=args.phase1_prompt_type,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        use_inverses=args.use_inverses,
        return_triplets=True,       # emit KB triples in the trace
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_length=args.max_completion_length,
    )

    answers, traces = agent.run(
        queries=[EXAMPLE_QUESTION],
        contexts=[EXAMPLE_CONTEXTS],
    )
    result = build_result(agent, answers, traces, args.model_path)

    _write_output(Path(args.output_path), result)
    print(f"Wrote {args.output_path}")
    result["_json_path"] = args.output_path   # for _print_summary hints only
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
