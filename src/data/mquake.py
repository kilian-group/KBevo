"""Loader for the pinned MQuAKE-Remastered CF6334 artifact."""

import os
import random
from typing import Any, Dict, Optional

import pyarrow.parquet as pq
from datasets import Dataset

from .provenance import hf_dataset_file

_SPLIT_KEY = "6334"


def _get_labels(example: Dict[str, Any]):
    split_col = example.get("split") or {}
    return split_col.get(_SPLIT_KEY) or []


def _normalize_example(example: Dict[str, Any], index: int, seed: int):
    questions = example.get("questions") or []
    if questions:
        example["question"] = random.Random(f"{seed}:{index}").choice(questions)
    else:
        example["question"] = ""
    example["answers"] = [example["answer"]] + list(example.get("answer_alias") or [])
    example["new_answers"] = [example["new_answer"]] + list(example.get("new_answer_alias") or [])
    labels = _get_labels(example)
    if "test_edited" in labels:
        example["mquake_split_type"] = "test_edited"
    elif "train_edited" in labels:
        example["mquake_split_type"] = "train_edited"
    else:
        example["mquake_split_type"] = "test_unedited"
    return example


def _resolve_path(source: str, mquake_path: Optional[str]) -> str:
    explicit = mquake_path or os.environ.get("MQUAKE_PATH")
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.exists(path):
            raise FileNotFoundError(f"MQuAKE parquet not found: {path}")
        return path
    if source == "local":
        raise FileNotFoundError(
            "source='local' requires mquake_path=... or the MQUAKE_PATH environment variable"
        )
    if source not in {"auto", "hf", "remote"}:
        raise ValueError(f"Unsupported MQuAKE source: {source}")
    return hf_dataset_file("mquake")


def load_mquake(
    split: str,
    limit: Optional[int],
    seed: Optional[int] = 42,
    source: str = "auto",
    mquake_path: Optional[str] = None,
):
    effective_seed = 42 if seed is None else seed
    path = _resolve_path(source, mquake_path)
    table = pq.read_table(path).replace_schema_metadata(None)
    raw_dataset = Dataset(table).shuffle(seed=effective_seed)

    if split == "train":
        subset = raw_dataset.filter(lambda ex: "train_edited" in _get_labels(ex))
    elif split == "test" or split.startswith("eval-"):
        subset = raw_dataset.filter(
            lambda ex: bool({"test_edited", "test_unedited"} & set(_get_labels(ex)))
        )
    else:
        subset = raw_dataset

    subset = subset.map(
        lambda example, index: _normalize_example(example, index, effective_seed),
        with_indices=True,
        load_from_cache_file=False,
    )
    selected_limit = len(subset) if limit is None else min(limit, len(subset))
    return subset.select(range(selected_limit))
