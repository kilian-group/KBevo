"""Script for loading HotpotQA dataset.

This module exposes `load_hotpotqa(setting, split, source="auto", limit=None)`
that returns a Hugging Face Dataset with a unified schema:

  - id: str
  - question: str
  - answers: List[str]
  - contexts: List[str]
  - supporting_facts: List[Dict[str, Any]]

Loading preference:
  - source="auto": prefer local raw files under data/raw/hotpotqa if available,
    otherwise fall back to Hugging Face datasets ("hotpot_qa").
  - source="local": only try local raw.
  - source="hf": only try Hugging Face datasets.

No caching is performed here.
"""

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore

from .provenance import hf_source


def _repo_root() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(this_dir)
    return os.path.abspath(os.path.join(src_dir, ".."))


def _local_hotpotqa_file(setting: str, split: str) -> Optional[str]:
    """Return path to local raw HotpotQA JSON for given setting/split if known."""
    split_norm = _normalize_split(split)
    fname: Optional[str] = None
    if setting == "distractor":
        if split_norm == "validation":
            fname = "hotpot_dev_distractor_v1.json"
        elif split_norm == "train":
            fname = "hotpot_train_v1.1.json"
        else:
            fname = None
    elif setting == "fullwiki":
        if split_norm == "validation":
            fname = "hotpot_dev_fullwiki_v1.json"
        elif split_norm == "test":
            fname = "hotpot_test_fullwiki_v1.json"
        else:
            fname = None
    if fname is None:
        return None
    path = os.path.join(_repo_root(), "data", "raw", "hotpotqa", fname)
    return path if os.path.exists(path) else None


def _normalize_split(split: str) -> str:
    if split.lower() in {"dev", "validation"}:
        return "validation"
    return split.lower()


def _build_contexts(context_field: Any) -> List[str]:
    """
    Build paragraph strings assuming the context is a dict:
      {"title": [t1, t2, ...], "sentences": [[sents1...], [sents2...], ...]}
    Titles and sentence lists are matched by index.
    """
    if not isinstance(context_field, dict):
        return []
    titles = context_field.get("title")
    sentences = context_field.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return []
    contexts: List[str] = []
    for i, title in enumerate(titles):
        sents_i = sentences[i] if i < len(sentences) else []
        sent_list = [s for s in (sents_i or [])]
        paragraph = f"{title}: " + " ".join(sent_list).strip()
        contexts.append(paragraph.strip())
    return contexts


def _build_supporting_facts(sf_field: Any) -> List[Dict[str, Any]]:
    if not sf_field:
        return []
    titles_and_ids = zip(sf_field['title'], sf_field['sent_id'])
    result: List[Dict[str, Any]] = []
    for item in titles_and_ids:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sent_id = item
        result.append({"title": str(title), "sentence_id": int(sent_id)})
    return result


def _normalize_examples_pylist(examples: List[Dict[str, Any]]) -> HFDataset:
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        ex_id = ex.get("_id") or ex.get("id") or ""
        question = ex.get("question") or ""
        answer = ex.get("answer")
        answers: List[str]
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = [str(a) for a in answer]
        else:
            answers = ex.get("answers") or []
            answers = [str(a) for a in answers]
        contexts = _build_contexts(ex.get("context"))
        supporting_facts = _build_supporting_facts(ex.get("supporting_facts"))
        rows.append(
            {
                "id": str(ex_id),
                "question": str(question),
                "answers": answers,
                "contexts": contexts,
                "supporting_facts": supporting_facts,
            }
        )
    return HFDataset.from_list(rows)


def _build_hotpotqa_rag_contexts_from_raw(examples: List[Dict[str, Any]]) -> List[str]:
    """Build a global RAG corpus from raw HotpotQA JSON examples."""
    contexts: List[str] = []
    for ex in examples:
        context_field = ex.get("context")
        if isinstance(context_field, dict):
            titles = context_field.get("title")
            sentences = context_field.get("sentences")
            if not isinstance(titles, list) or not isinstance(sentences, list):
                continue
            for i, title in enumerate(titles):
                sents_i = sentences[i] if i < len(sentences) else []
                sent_list = [s for s in (sents_i or [])]
                paragraph = f"{title}: " + " ".join(sent_list).strip()
                paragraph = paragraph.strip()
                if paragraph:
                    contexts.append(paragraph)
            continue
        if isinstance(context_field, list):
            for item in context_field:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                title, sents_i = item
                if not isinstance(sents_i, list):
                    continue
                sent_list = [s for s in (sents_i or [])]
                paragraph = f"{title}: " + " ".join(sent_list).strip()
                paragraph = paragraph.strip()
                if paragraph:
                    contexts.append(paragraph)
    return contexts


def _build_hotpotqa_rag_records_from_raw(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build structured corpus records from raw HotpotQA JSON examples."""
    records: List[Dict[str, Any]] = []
    for ex in examples:
        context_field = ex.get("context")
        if isinstance(context_field, dict):
            titles = context_field.get("title")
            sentences = context_field.get("sentences")
            if not isinstance(titles, list) or not isinstance(sentences, list):
                continue
            for i, title in enumerate(titles):
                sents_i = sentences[i] if i < len(sentences) else []
                sent_list = [s for s in (sents_i or [])]
                paragraph = " ".join(sent_list).strip()
                records.append(
                    {
                        "title": str(title).strip(),
                        "contents": paragraph,
                    }
                )
            continue
        if isinstance(context_field, list):
            for item in context_field:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                title, sents_i = item
                if not isinstance(sents_i, list):
                    continue
                sent_list = [s for s in (sents_i or [])]
                paragraph = " ".join(sent_list).strip()
                records.append(
                    {
                        "title": str(title).strip(),
                        "contents": paragraph,
                    }
                )
    return records


def _dedupe_paragraph_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by (title, contents) while preserving order."""
    seen = set()
    output: List[Dict[str, Any]] = []
    for record in records:
        title = str(record.get("title", "")).strip()
        contents = str(record.get("contents", "")).strip()
        if not title and not contents:
            continue
        key = (title, contents)
        if key in seen:
            continue
        seen.add(key)
        output.append({"title": title, "contents": contents})
    return output


def _dedupe_nonempty_paragraphs(paragraphs: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for paragraph in paragraphs:
        text = str(paragraph).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _normalize_hotpotqa_corpus_record(record: Any) -> Optional[Dict[str, Any]]:
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
        title, article = _split_title_article(record)
        return {"title": title, "contents": article}
    return None


def load_hotpotqa_rag_corpus(path: str) -> List[Dict[str, Any]]:
    """Load and build a deduplicated HotpotQA RAG corpus from a JSON/JSONL file."""
    _, ext = os.path.splitext(path)
    records: List[Dict[str, Any]] = []
    if ext.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                coerced = _normalize_hotpotqa_corpus_record(record)
                if coerced:
                    records.append(coerced)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError(f"Unexpected HotpotQA JSON format at {path}")
        records = _build_hotpotqa_rag_records_from_raw(data)

    deduped = _dedupe_paragraph_records(records)
    for idx, record in enumerate(deduped):
        record["id"] = idx
    return deduped


def _build_golden_contexts(context_field: Any, sf_field: Any) -> List[str]:
    """Build golden context strings - only contexts whose title is in supporting_facts."""
    if not isinstance(context_field, dict) or not sf_field:
        return []

    # Get supporting titles
    supporting_titles = set(sf_field.get('title', []))

    titles = context_field.get("title")
    sentences = context_field.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return []

    golden: List[str] = []
    for i, title in enumerate(titles):
        if title not in supporting_titles:
            continue
        sents_i = sentences[i] if i < len(sentences) else []
        sent_list = [s for s in (sents_i or [])]
        paragraph = f"{title}: " + " ".join(sent_list).strip()
        golden.append(paragraph.strip())
    return golden


def _normalize_hf_dataset(ds: HFDataset) -> HFDataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex_id = ex.get("_id") or ex.get("id") or ""
        answer = ex.get("answer")
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = [str(a) for a in answer]
        else:
            answers = [str(a) for a in ex.get("answers", [])]
        return {
            "id": str(ex_id),
            "question": str(ex.get("question", "")),
            "answers": answers,
            "contexts": _build_contexts(ex.get("context")),
            "golden_contexts": _build_golden_contexts(ex.get("context"), ex.get("supporting_facts")),
            "supporting_facts": _build_supporting_facts(ex.get("supporting_facts")),
        }

    # In distributed runs, multiple ranks may call this simultaneously. Avoid shared on-disk
    # map cache writes, which can race and fail with FileNotFoundError in shutil.move.
    return ds.map(
        _map,
        remove_columns=ds.column_names,
        desc="normalize hotpotqa",
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def load_hotpotqa(
    setting: str,
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    sub_split: Optional[str] = None,
) -> HFDataset:
    """Load HotpotQA with unified schema.

    Args:
        setting: "distractor" or "fullwiki".
        split: "train", "dev"/"validation", or "test" (where available).
        source: "auto" (prefer local), "local", or "hf".
        limit: optional max number of rows to return.
        seed: optional random seed for shuffling. If provided, dataset will be shuffled deterministically.
    """

    split_norm = _normalize_split(split)

    # Try local raw
    if source in ("auto", "local"):
        local_path = _local_hotpotqa_file(setting, split_norm)
        if local_path is not None:
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                ds = _normalize_examples_pylist(data)
                # Shuffle with seed if provided
                if seed is not None:
                    ds = ds.shuffle(seed=seed)
                if limit is not None:
                    ds = ds.select(range(min(limit, len(ds))))
                return ds
        if source == "local":
            raise FileNotFoundError(
                f"Local HotpotQA file not found for setting={setting} split={split_norm}"
            )

    # Fallback to Hugging Face
    hf_split = split_norm
    pinned_source = hf_source("hotpotqa")
    try:
        raw = load_dataset(
            pinned_source["path"], setting, split=hf_split, revision=pinned_source["revision"]
        )  # type: ignore
    except Exception as e:
        # Common failure mode: stale/incompatible cached dataset metadata across datasets versions.
        # Retry with forced redownload, then with a fresh isolated cache directory.
        try:
            raw = load_dataset(
                pinned_source["path"],
                setting,
                split=hf_split,
                revision=pinned_source["revision"],
                download_mode="force_redownload",
            )  # type: ignore
        except Exception:
            try:
                isolated_cache_dir = os.path.join(tempfile.gettempdir(), "hf_datasets_hotpotqa_clean_cache")
                os.makedirs(isolated_cache_dir, exist_ok=True)
                raw = load_dataset(
                    pinned_source["path"],
                    setting,
                    split=hf_split,
                    revision=pinned_source["revision"],
                    cache_dir=isolated_cache_dir,
                    download_mode="force_redownload",
                )  # type: ignore
            except Exception:
                raise RuntimeError(
                    f"Failed to load HotpotQA from Hugging Face (setting={setting}, split={hf_split}): {e}"
                )
    ds = _normalize_hf_dataset(raw)
    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)
        
    if sub_split is not None:
        MAGIC_START_IDX = 82347
        MAGIC_TRAIN_MAX_SIZE = 8000
        MAGIC_VAL_MAX_SIZE   = 100

        n = len(ds)
        assert MAGIC_START_IDX + MAGIC_TRAIN_MAX_SIZE + MAGIC_VAL_MAX_SIZE == len(ds)
        
        if sub_split == "train":
            if limit is not None and limit <= MAGIC_TRAIN_MAX_SIZE:
                ds = ds.select(range(n - MAGIC_VAL_MAX_SIZE - limit, n - MAGIC_VAL_MAX_SIZE))
            else:
                ds = ds.select(range(n - MAGIC_VAL_MAX_SIZE))
        if sub_split == "eval":
            assert limit  <= MAGIC_VAL_MAX_SIZE
            ds = ds.select(range(n - limit, n))

        ds = ds.select(range(min(limit, len(ds))))
        
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
