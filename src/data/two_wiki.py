"""Portable 2WikiMultiHopQA validation loader."""

import json
import os
from typing import Optional

from datasets import Dataset as HFDataset  # type: ignore

from .hotpotqa import _normalize_split
from .provenance import hf_dataset_file


def _resolve_path(source: str, two_wiki_path: Optional[str]) -> str:
    explicit = two_wiki_path or os.environ.get("TWO_WIKI_PATH")
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.exists(path):
            raise FileNotFoundError(f"2Wiki dev file not found: {path}")
        return path
    if source == "local":
        raise FileNotFoundError(
            "source='local' requires two_wiki_path=... or the TWO_WIKI_PATH environment variable"
        )
    if source not in {"auto", "hf", "remote"}:
        raise ValueError(f"Unsupported 2Wiki source: {source}")
    return hf_dataset_file("2wiki")


def load_2wiki(
    setting: str,
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    two_wiki_path: Optional[str] = None,
) -> HFDataset:
    del setting  # 2Wiki exposes a single distractor-style validation file here.
    split = _normalize_split(split)
    if split != "validation":
        raise NotImplementedError("Use the dev/validation split for 2WikiMultiHopQA")

    with open(_resolve_path(source, two_wiki_path), "r", encoding="utf-8") as stream:
        data = json.load(stream)
    dataset = _normalize_2wiki_data(data)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def _normalize_2wiki_data(data):
    rows = []
    for ex in data:
        ex_id = ex.get("_id") or ex.get("id") or ""
        answer = ex.get("answer")
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = [str(value) for value in answer]
        else:
            answers = [str(value) for value in (ex.get("answers") or [])]
        contexts = _build_contexts(ex.get("context"))
        supporting_facts = _build_supporting_facts(ex.get("supporting_facts"))
        rows.append(
            {
                "id": str(ex_id),
                "question": str(ex.get("question") or ""),
                "answers": answers,
                "contexts": contexts,
                "supporting_facts": supporting_facts,
                "golden_contexts": _build_golden_contexts(
                    ex.get("context"), ex.get("supporting_facts")
                ),
            }
        )
    return HFDataset.from_list(rows)


def _build_contexts(context):
    if not context:
        return []
    return [
        f"{item[0]}: " + " ".join(item[1]).strip()
        for item in context
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]


def _build_supporting_facts(supporting_facts):
    if not supporting_facts:
        return []
    return [
        {"title": str(item[0]), "sentence_id": int(item[1])}
        for item in supporting_facts
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]


def _build_golden_contexts(context, supporting_facts):
    if not context or not supporting_facts:
        return []
    supporting_titles = {
        item[0] for item in supporting_facts if isinstance(item, (list, tuple)) and item
    }
    return [
        f"{item[0]}: " + " ".join(item[1]).strip()
        for item in context
        if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] in supporting_titles
    ]
