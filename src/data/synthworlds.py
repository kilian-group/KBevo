"""Script for loading SynthWorlds dataset.

This module exposes `load_synthworlds(subset, split, source="hf", limit=None, seed=None)`
that returns a Hugging Face Dataset with a unified schema:

  - id: str
  - question: str
  - answers: List[str]
  - golden_contexts: List[str]
  - contexts: List[str] (same as golden_contexts)
  - supporting_facts: List[Dict[str, Any]] (empty - not used for SynthWorlds)

Note: SynthWorlds 'contexts' field is set to the same value as 'golden_contexts'.

Dataset info:
  - HuggingFace: kenqgu/SynthWorlds
  - Subsets: 'qa-sm' (small), 'qa-rm' (regular multi-hop)
  - Split: 'test' only (mapped from 'dev'/'validation' input)
"""

from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore

from .provenance import hf_source


def _normalize_split(split: str) -> str:
    """Normalize split name. SynthWorlds only has 'test' split."""
    if split.lower() in {"dev", "validation"}:
        return "test"
    if split.lower() == "test":
        return "test"
    raise ValueError(
        f"SynthWorlds only supports 'dev'/'validation'/'test' splits, got '{split}'. "
        "All map to the 'test' split."
    )


def _normalize_subset(subset: str) -> str:
    """Normalize subset name."""
    subset_norm = subset.lower().strip()
    if subset_norm in {"qa-sm", "sm", "small"}:
        return "qa-sm"
    if subset_norm in {"qa-rm", "rm", "regular"}:
        return "qa-rm"
    raise ValueError(
        f"SynthWorlds only supports 'qa-sm' or 'qa-rm' subsets, got '{subset}'"
    )


def _normalize_hf_dataset(hf_dataset: Any) -> HFDataset:
    """Convert SynthWorlds HF dataset to unified schema.

    SynthWorlds fields:
      - instance_id → id
      - query → question
      - gold_answers → answers
      - gold_docs → golden_contexts (also copied to contexts)
    """
    rows: List[Dict[str, Any]] = []

    for ex in hf_dataset:
        ex_id = ex.get("instance_id", "")
        question = ex.get("query", "")

        # gold_answers is a list of strings
        gold_answers = ex.get("gold_answers", [])
        answers: List[str] = [str(a) for a in gold_answers] if isinstance(gold_answers, list) else []

        # gold_docs is a list of strings (no titles, just text)
        gold_docs = ex.get("gold_docs", [])
        golden_contexts: List[str] = [str(doc) for doc in gold_docs] if isinstance(gold_docs, list) else []

        rows.append({
            "id": str(ex_id),
            "question": str(question),
            "answers": answers,
            "golden_contexts": golden_contexts,
            "contexts": golden_contexts,  # Same as golden_contexts for SynthWorlds
            # Supporting facts not used for SynthWorlds (no title/sentence granularity)
            "supporting_facts": [],
        })

    return HFDataset.from_list(rows)


def load_synthworlds(
    subset: str,
    split: str,
    source: str = "hf",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> HFDataset:
    """Load SynthWorlds with unified schema.

    Args:
        subset: "qa-sm" (small) or "qa-rm" (regular multi-hop)
        split: "dev", "validation", or "test" (all map to 'test' split)
        source: "hf" only (dataset only available on HuggingFace)
        limit: optional maximum number of rows
        seed: optional random seed for shuffling

    Returns:
        HFDataset with unified schema ('contexts' is set to same value as 'golden_contexts')
    """
    subset_norm = _normalize_subset(subset)
    split_norm = _normalize_split(split)

    if source not in ("auto", "hf"):
        raise ValueError(f"SynthWorlds only supports source='hf', got '{source}'")

    try:
        pinned_source = hf_source("synthworlds")
        raw = load_dataset(
            pinned_source["path"],
            subset_norm,
            split=split_norm,
            revision=pinned_source["revision"],
        )  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SynthWorlds from HuggingFace (subset={subset_norm}, split={split_norm}): {e}"
        )

    ds = _normalize_hf_dataset(raw)

    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return ds
