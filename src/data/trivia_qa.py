"""Loader for TriviaQA dataset with unified schema.

Exposes `load_trivia_qa(split, source="auto", limit=None)` returning an HF Dataset
with fields:

- id: str
- question: str
- answers: List[str]
- contexts: List[str]
- context_titles: List[str]  (parallel to contexts, for splitting logic)
- supporting_facts: List[Dict[str, Any]]  (empty for TriviaQA)
- golden_contexts: List[str]

Notes:
- TriviaQA does not provide supporting facts, so supporting_facts is always empty.
- Contexts are formatted as "title: wiki_context" for each entity page.
- context_titles stores the titles separately to avoid fragile string parsing during splitting.
- Context splitting (sentence grouping >= 800 chars) is handled by eval_multihop.py, not here.
"""

import os
from typing import Any, Dict, List, Optional
from venv import logger

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore

from .provenance import hf_source


def _normalize_split(split: str) -> str:
    """Normalize split names (dev -> validation)."""
    s = split.lower()
    if s in {"dev", "validation"}:
        return "validation"
    return s


def _build_context_titles(entity_pages: Any) -> List[str]:
    """Extract title list from entity_pages.

    Args:
        entity_pages: Dict with 'title' (list) and 'wiki_context' (list) keys

    Returns:
        List of titles (parallel to contexts returned by _build_contexts)
    """
    if not isinstance(entity_pages, dict):
        return []

    titles = entity_pages.get("title", [])
    wiki_contexts = entity_pages.get("wiki_context", [])

    if not isinstance(titles, list) or not isinstance(wiki_contexts, list):
        return []

    # Return only titles that have corresponding wiki_contexts
    # This ensures context_titles aligns with contexts list
    return [str(title) for title in titles[:len(wiki_contexts)]]


def _build_contexts(entity_pages: Any) -> List[str]:
    """Build context strings from entity_pages.

    Args:
        entity_pages: Dict with 'title' (list) and 'wiki_context' (list) keys

    Returns:
        List of wiki_context strings (without title prefix - title is stored separately in context_titles)
    """
    if not isinstance(entity_pages, dict):
        return []

    titles = entity_pages.get("title", [])
    wiki_contexts = entity_pages.get("wiki_context", [])

    if not isinstance(titles, list) or not isinstance(wiki_contexts, list):
        return []

    # Return only wiki_contexts that have corresponding titles
    # The title will be added during splitting using the context_titles field
    return [str(wiki_context).strip() for wiki_context in wiki_contexts[:len(titles)] if str(wiki_context).strip()]


def _build_answers(answer_dict: Any) -> List[str]:
    """Build answer list from TriviaQA answer dict.

    Args:
        answer_dict: Dict with 'aliases' and 'normalized_aliases' keys

    Returns:
        List combining all answer aliases
    """
    if not isinstance(answer_dict, dict):
        return []

    answers: List[str] = []

    # Combine normalized and unnormalized aliases
    normalized = answer_dict.get("normalized_aliases", [])
    unnormalized = answer_dict.get("aliases", [])

    if isinstance(normalized, list):
        answers.extend([str(a) for a in normalized])
    if isinstance(unnormalized, list):
        answers.extend([str(a) for a in unnormalized])

    return answers


def _normalize_hf_dataset(ds: HFDataset) -> HFDataset:
    """Normalize TriviaQA HF dataset to unified schema."""
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex_id = ex.get("question_id") or ex.get("id") or ""
        question = ex.get("question", "")

        # Build answers from the answer dict
        answers = _build_answers(ex.get("answer"))

        # Build contexts and titles from entity_pages
        entity_pages = ex.get("entity_pages")
        contexts = _build_contexts(entity_pages)
        context_titles = _build_context_titles(entity_pages)

        return {
            "id": str(ex_id),
            "question": str(question),
            "answers": answers,
            "contexts": contexts,
            "context_titles": context_titles,  # Parallel list of titles for splitting logic
            "golden_contexts": contexts,  # No ground truth supporting docs, so use all
            "supporting_facts": [],  # TriviaQA has no sentence-level supporting facts
        }

    # Map to unified schema, avoid cache conflicts
    return ds.map(
        _map,
        remove_columns=ds.column_names,
        desc="normalize trivia_qa",
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def load_trivia_qa(
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    setting: Optional[str] = None,
) -> HFDataset:
    """Load TriviaQA with unified schema.

    Args:
        split: "train", "dev"/"validation", or "test"
        source: "auto", "local", or "hf" (only "hf" supported for TriviaQA)
        limit: optional max number of rows to return
        seed: optional random seed for shuffling
        setting: dataset setting (default: "rc.wikipedia")

    Returns:
        HFDataset with unified schema
    """
    split_norm = _normalize_split(split)

    # TriviaQA setting (rc.wikipedia is the Wikipedia-based reading comprehension version)
    if setting is not None and setting != "rc.wikipedia":
        logger.info(f"Warning: Unsupported setting '{setting}' for TriviaQA. Defaulting to 'rc.wikipedia'.")
    trivia_setting = "rc.wikipedia"

    # Load from Hugging Face
    try:
        pinned_source = hf_source("trivia_qa")
        raw = load_dataset(
            pinned_source["path"],
            trivia_setting,
            split=split_norm,
            revision=pinned_source["revision"],
        )  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TriviaQA from Hugging Face (setting={trivia_setting}, split={split_norm}): {e}"
        )

    # Normalize to unified schema
    ds = _normalize_hf_dataset(raw)

    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    # Limit results if requested
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return ds
