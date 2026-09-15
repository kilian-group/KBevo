"""Loader for MuSiQue dataset with unified schema.

Exposes `load_musique(split, source="auto", limit=None)` returning an HF Dataset
with fields:

- id: str
- question: str
- answers: List[str]
- contexts: List[str]
- supporting_facts: List[Dict[str, Any]]

Notes:
- MuSiQue provides paragraph-level `is_supporting` flags but not sentence spans.
  We encode each supporting paragraph as `{title: <title>, sentence_id: 0}`.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore

from .provenance import hf_source


def _normalize_split(split: str) -> str:
    s = split.lower()
    if s in {"dev", "validation"}:
        return "validation"
    return s


def _build_contexts(paragraphs: Any) -> List[str]:
    if not paragraphs:
        return []
    result: List[str] = []
    for p in paragraphs:
        if not isinstance(p, dict):
            # best-effort stringify
            result.append(str(p))
            continue
        title = str(p.get("title", "")).strip()
        text = str(p.get("paragraph_text", "")).strip()
        if title:
            result.append(f"{title}: {text}".strip())
        else:
            result.append(text)
    return result


def _build_supporting_facts(paragraphs: Any) -> List[Dict[str, Any]]:
    if not paragraphs:
        return []
    facts: List[Dict[str, Any]] = []
    for p in paragraphs:
        if isinstance(p, dict) and p.get("is_supporting"):
            facts.append({"title": str(p.get("title", "")), "sentence_id": 0})
    return facts


def _dedupe_nonempty_paragraphs(paragraphs: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for paragraph in paragraphs:
        text = str(paragraph).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _build_musique_rag_contexts_from_raw(examples: List[Dict[str, Any]]) -> List[str]:
    """Build a global RAG corpus from raw MuSiQue JSON examples."""
    contexts: List[str] = []
    for ex in examples:
        contexts.extend(_build_contexts(ex.get("paragraphs")))
    return contexts


def _build_musique_rag_records_from_raw(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build structured corpus records from raw MuSiQue JSON examples."""
    records: List[Dict[str, Any]] = []
    for ex in examples:
        paragraphs = ex.get("paragraphs") or []
        for p in paragraphs:
            if not isinstance(p, dict):
                continue
            title = str(p.get("title", "")).strip()
            text = str(p.get("paragraph_text", "")).strip()
            records.append({"title": title, "contents": text})
    return records


def _dedupe_paragraph_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by (title, contents) while preserving order."""
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for record in records:
        title = str(record.get("title", "")).strip()
        contents = str(record.get("contents", "")).strip()
        if not title and not contents:
            continue
        key = (title, contents)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"title": title, "contents": contents})
    return deduped


def _build_golden_contexts(paragraphs: Any) -> List[str]:
    """Build golden context strings - only contexts marked as is_supporting."""
    if not paragraphs:
        return []
    result: List[str] = []
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        if not p.get("is_supporting"):
            continue
        title = str(p.get("title", "")).strip()
        text = str(p.get("paragraph_text", "")).strip()
        if title:
            result.append(f"{title}: {text}".strip())
        else:
            result.append(text)
    return result


def _normalize_hf_dataset(ds: HFDataset) -> HFDataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex_id = ex.get("id") or ex.get("_id") or ""
        # MuSiQue usually has a single string answer; still normalize to list
        answer_field = ex.get("answer")
        if isinstance(answer_field, list):
            answers = [str(a) for a in answer_field]
        elif answer_field is None:
            answers = [str(a) for a in ex.get("answers", [])]
        else:
            answers = [str(answer_field)]
        paragraphs = ex.get("paragraphs")
        return {
            "id": str(ex_id),
            "question": str(ex.get("question", "")),
            "answers": answers,
            "contexts": _build_contexts(paragraphs),
            "golden_contexts": _build_golden_contexts(paragraphs),
            "supporting_facts": _build_supporting_facts(paragraphs),
        }

    return ds.map(_map, remove_columns=ds.column_names, desc="normalize musique")


def load_musique(
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> HFDataset:
    """Load MuSiQue with unified schema.

    Args:
        split: "train", "dev"/"validation", or "test" (if available).
        source: "auto" or "hf" (HF only at the moment).
        limit: optional maximum number of rows.
        seed: optional random seed for shuffling. If provided, dataset will be shuffled deterministically.
    """
    split_norm = _normalize_split(split)

    if source not in ("auto", "hf"):
        raise ValueError(f"Unsupported source for MuSiQue: {source}")

    try:
        pinned_source = hf_source("musique")
        raw = load_dataset(
            pinned_source["path"], split=split_norm, revision=pinned_source["revision"]
        )  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to load MuSiQue from Hugging Face (split={split_norm}): {e}")

    ds = _normalize_hf_dataset(raw)
    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _normalize_musique_corpus_record(record: Any) -> Optional[Dict[str, Any]]:
    """Normalize various record shapes into {title, contents}."""
    if isinstance(record, dict):
        title = str(record.get("title", "")).strip()
        contents = record.get("contents")
        if contents is None:
            contents = record.get("context")
        if contents is None:
            contents = record.get("paragraph_text", "")
        return {"title": title, "contents": str(contents).strip()}
    if isinstance(record, str):
        if ": " in record:
            title, contents = record.split(": ", 1)
            return {"title": title.strip(), "contents": contents.strip()}
        return {"title": "", "contents": record.strip()}
    return None


def load_musique_rag_corpus(path: str) -> List[Dict[str, Any]]:
    """Load and build a deduplicated MuSiQue RAG corpus from a JSON/JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MuSiQue RAG corpus file not found: {path}")

    _, ext = os.path.splitext(path)
    records: List[Dict[str, Any]] = []
    if ext.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                coerced = _normalize_musique_corpus_record(record)
                if coerced:
                    records.append(coerced)
                    continue
                if isinstance(record, dict) and "paragraphs" in record:
                    records.extend(_build_musique_rag_records_from_raw([record]))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            records = data.get("data") or data.get("examples") or []
        else:
            records = data

    if not isinstance(records, list):
        return []
    if records and all(isinstance(item, str) for item in records):
        coerced = [_normalize_musique_corpus_record(item) for item in records]
        deduped = _dedupe_paragraph_records([c for c in coerced if c])
    elif records and all(
        isinstance(item, dict) and ("title" in item or "contents" in item) for item in records
    ):
        coerced: List[Dict[str, Any]] = []
        for item in records:
            record = _normalize_musique_corpus_record(item)
            if record:
                coerced.append(record)
        deduped = _dedupe_paragraph_records(coerced)
    else:
        deduped = _dedupe_paragraph_records(_build_musique_rag_records_from_raw(records))

    for idx, record in enumerate(deduped):
        record["id"] = idx
    return deduped


def load_musique_rag_corpus_from_hf(
    split: str,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a deduplicated MuSiQue RAG corpus from the HF dataset."""
    split_norm = _normalize_split(split)
    pinned_source = hf_source("musique")
    raw = load_dataset(
        pinned_source["path"], split=split_norm, revision=pinned_source["revision"]
    )  # type: ignore
    if seed is not None:
        raw = raw.shuffle(seed=seed)
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))
    records = _build_musique_rag_records_from_raw(list(raw))
    return _dedupe_paragraph_records(records)


def write_musique_rag_corpus_jsonl(
    path: str,
    split: str,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> int:
    """Write a MuSiQue RAG corpus JSONL file from the HF dataset."""
    records = load_musique_rag_corpus_from_hf(split=split, limit=limit, seed=seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            json.dump(
                {
                    "id": idx,
                    "title": record.get("title", ""),
                    "contents": record.get("contents", ""),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    return len(records)
