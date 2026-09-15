"""Utilities to evaluate prediction JSONs produced by scripts/run_agent.py.

Expected file structure: preds/{type}/{dataset}_{setting}/{model}/{split}_seed{s}_bn={n}_bs={b}.json

Reads a preds JSON with deduplicated metadata at the top level:
{
  "metadata": { model, split, batch_size, batch_number, type, seed, retrieval },
  "inference_params": { seed, temperature, max_tokens },
  "results": {
    "qid": { pred, gold_answer, gold_evidence, question, trace, evidence }
  }
}

Computes answer-only metrics (EM, F1, precision, recall) and aggregates.
This module exposes pure functions so a thin CLI can wrap it.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

from .metrics import exact_match_score, f1_score, normalize_answer, exact_match_relaxed, mquake_f1_score
from data import get_dataset


@dataclass
class EvalMeta:
    dataset: str
    setting: str | None
    agent: str
    llm: str
    bn: int
    bs: int
    split: str | None
    preds_path: str


def _safe_join_gold(gold: Any) -> str:
    """Convert a gold answer field into a scoring string.

    - If list-like, join with newline as requested.
    - Else, cast to string.
    """
    if gold is None:
        return ""
    if isinstance(gold, (list, tuple)):
        return "\n".join(str(x) for x in gold)
    return str(gold)


def _filename_parts_from_path(path: str) -> Tuple[str | None, str | None, str | None, int | None, int | None]:
    """Parse dataset, setting, split, bn, bs from a preds path.

    Expected pattern: type/dataset_setting/model/split_seed={s}_bn={n}_bs={b}.json
    Returns (dataset, setting, split, bn, bs) with None on failure.
    """
    name = os.path.basename(path)
    
    # Parse filename: split_seed={s}_bn={n}_bs={b}.json
    m = re.match(r"^(?P<split>[^_]+)_seed=(?P<seed>\d+)_bn=(?P<bn>\d+)_bs=(?P<bs>\d+)\.json$", name)
    if not m:
        return None, None, None, None, None
    
    # Extract dataset and setting from directory path
    parts = os.path.normpath(path).split(os.sep)
    try:
        idx = parts.index("preds")
        if len(parts) > idx + 2:
            # parts[idx+1] is type (icl/rag/db)
            # parts[idx+2] is dataset_setting
            dataset_setting = parts[idx + 2]
            # Split on first underscore to get dataset and setting
            if "_" in dataset_setting:
                dataset, setting = dataset_setting.split("_", 1)
                return (
                    dataset,
                    setting,
                    m.group("split"),
                    int(m.group("bn")),
                    int(m.group("bs")),
                )
    except (ValueError, IndexError):
        pass
    
    return None, None, None, None, None


def _extract_meta(preds: Dict[str, Any], preds_path: str) -> EvalMeta:
    """Extract metadata from the top-level metadata field and path as fallback."""
    # Defaults from path parsing
    dataset, setting, split, bn_from_name, bs_from_name = _filename_parts_from_path(preds_path)
    agent = None
    llm = None
    bn = None
    bs = None

    # Extract from top-level metadata if present
    meta = preds.get("metadata", {}) or {}
    if meta:
        agent = meta.get("type")
        llm = meta.get("model")
        split = split or meta.get("split")
        bn = meta.get("batch_number")
        bs = meta.get("batch_size")

    # Parse directory structure: preds/type/dataset_setting/model/filename.json
    parts = os.path.normpath(preds_path).split(os.sep)
    try:
        idx = parts.index("preds")
        if len(parts) > idx + 1:
            # parts[idx+1] is type (icl/rag/db)
            if agent is None:
                agent = parts[idx + 1]
            # parts[idx+3] is model (if using new structure)
            if len(parts) > idx + 3 and llm is None:
                llm = parts[idx + 3]
    except (ValueError, IndexError):
        pass

    # Fallback: path-derived agent if still not present
    if agent is None:
        agent = "unknown"

    # Prefer metadata values; fall back to filename
    if bn is None:
        bn = bn_from_name
    if bs is None:
        bs = bs_from_name

    return EvalMeta(
        dataset=dataset or "unknown",
        setting=str(setting) if setting is not None else None,
        agent=str(agent or "unknown"),
        llm=str(llm or "unknown"),
        bn=int(bn) if isinstance(bn, int) or (isinstance(bn, str) and bn.isdigit()) else -1,
        bs=int(bs) if isinstance(bs, int) or (isinstance(bs, str) and bs.isdigit()) else -1,
        split=str(split) if split is not None else None,
        preds_path=preds_path,
    )


def evaluate_file(
    preds_path: str,
    dataset: Optional[str] = None,
    setting: Optional[str] = None,
    split: Optional[str] = None,
    source: str = "hf",
) -> Dict[str, Any]:
    """Evaluate a single preds JSON file and return metrics + metadata.

    For MQuAKE, answer_type is auto-inferred from the split name:
    splits starting with 'eval-edit' use new_answer, all others use answer.

    Returns:
        dict with keys:
          - metrics: { count, em, f1, precision, recall }
          - meta: EvalMeta as dict (dataset, agent, llm, bn, bs, split, preds_path)
    """
    with open(preds_path, "r", encoding="utf-8") as f:
        preds: Dict[str, Any] = json.load(f)

    meta = _extract_meta(preds, preds_path)

    # Determine dataset/setting/split using overrides when provided
    dataset_name = dataset or meta.dataset
    setting_name = setting or meta.setting
    split_name = split or meta.split

    # Determine answer_type from split: eval-edit* → new_answer, else → answer
    is_mquake = dataset_name and dataset_name.lower() in ("mquake", "mquake-remastered")
    answer_type = "new_answer" if (is_mquake and split_name and split_name.startswith("eval-edit")) else "answer"

    # Lazily loaded mapping from qid -> list of gold answers
    gold_by_id: Dict[str, list] | None = None

    total = 0
    sum_em = 0.0
    sum_f1 = 0.0
    sum_prec = 0.0
    sum_recall = 0.0
    retrieval_total_gold = 0
    retrieval_total_retrieved = 0
    retrieval_total_overlap = 0

    # Per-type accumulators for MQuAKE (test_edited / train_edited / test_unedited)
    type_sums: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0, "count": 0}
    )

    # Extract results from new format (with fallback for potential edge cases)
    results = preds.get("results", preds)

    for _qid, rec in results.items():
        pred_text = rec.get("pred", "")

        if is_mquake:
            # For MQuAKE, pick gold based on the split-level answer_type first:
            #   eval-original (answer_type=="answer") → always use gold_answer (original answers)
            #   eval-edit* (answer_type=="new_answer") → use per-example mquake_split_type:
            #     test_unedited → gold_answer; test_edited / train_edited → new_gold_answer
            if answer_type == "answer":
                gold_field = rec.get("gold_answer")
            else:
                mquake_type = rec.get("mquake_split_type")
                if mquake_type == "test_unedited":
                    gold_field = rec.get("gold_answer")
                else:
                    # Prefer new_gold_answer; fall back to gold_answer if missing
                    gold_field = rec.get("new_gold_answer") or rec.get("gold_answer")
        else:
            gold_field = (
                rec.get("gold_answer")
                if "gold_answer" in rec
                else rec.get("answers")
                if "answers" in rec
                else rec.get("true")
            )

        # Ensure gold_field is a list of possible answers
        if gold_field is None:
            gold_answers = []
        elif isinstance(gold_field, (list, tuple)):
            gold_answers = [str(x) for x in gold_field]
        else:
            gold_answers = [str(gold_field)]

        # If gold missing, try to load from dataset once
        if not gold_answers and dataset_name not in (None, "unknown") and split_name is not None:
            if gold_by_id is None:
                # get_dataset requires a setting param; for datasets without setting, pass a placeholder
                effective_setting = setting_name or "na"
                try:
                    ds = get_dataset(name=dataset_name, setting=effective_setting, split=split_name, source=source)
                    tmp: Dict[str, list] = {}
                    for row in ds:
                        ans = row.get("answers") or []
                        if isinstance(ans, list):
                            tmp[row["id"]] = [str(x) for x in ans]
                        else:
                            tmp[row["id"]] = [str(ans)]
                    gold_by_id = tmp
                except Exception:
                    gold_by_id = {}
            gold_answers = gold_by_id.get(str(_qid), [])

        if not gold_answers:
            continue

        # For multiple gold answers, check if pred matches ANY (for EM) and take MAX (for F1)
        em = 0.0
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0

        for gold_answer in gold_answers:
            if exact_match_score(pred_text, gold_answer):
                em = 1.0

            f1, precision, recall = f1_score(pred_text, gold_answer)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall

        f1, precision, recall = best_f1, best_precision, best_recall

        sum_em += em
        sum_f1 += f1
        sum_prec += precision
        sum_recall += recall
        total += 1

        # Accumulate per-type stats for MQuAKE
        if is_mquake:
            t = rec.get("mquake_split_type", "unknown")
            type_sums[t]["em"] += em
            type_sums[t]["f1"] += f1
            type_sums[t]["prec"] += precision
            type_sums[t]["recall"] += recall
            type_sums[t]["count"] += 1

        retrieval = rec.get("retrieval")
        if isinstance(retrieval, dict):
            gold_total = retrieval.get("gold_total")
            retrieved_total = retrieval.get("retrieved_total")
            overlap = retrieval.get("overlap")
            if isinstance(gold_total, int):
                retrieval_total_gold += gold_total
            if isinstance(retrieved_total, int):
                retrieval_total_retrieved += retrieved_total
            if isinstance(overlap, int):
                retrieval_total_overlap += overlap

    metrics = {
        "count": total,
        "em": round(sum_em / total, 4) if total else 0.0,
        "f1": round(sum_f1 / total, 4) if total else 0.0,
        "precision": round(sum_prec / total, 4) if total else 0.0,
        "recall": round(sum_recall / total, 4) if total else 0.0,
    }

    # Build per-type metrics for MQuAKE (test_edited / train_edited / test_unedited)
    per_type_metrics: Dict[str, Any] = {}
    if is_mquake and type_sums:
        for t, s in type_sums.items():
            c = s["count"]
            per_type_metrics[t] = {
                "count": c,
                "em": round(s["em"] / c, 4) if c else 0.0,
                "f1": round(s["f1"] / c, 4) if c else 0.0,
                "precision": round(s["prec"] / c, 4) if c else 0.0,
                "recall": round(s["recall"] / c, 4) if c else 0.0,
            }

    output: Dict[str, Any] = {
        "metrics": metrics,
        "meta": {
            "dataset": meta.dataset,
            "setting": meta.setting,
            "agent": meta.agent,
            "llm": meta.llm,
            "bn": meta.bn,
            "bs": meta.bs,
            "split": meta.split,
            "preds_path": meta.preds_path,
            # Pass through extra fields from preds metadata for reproducibility
            **{k: v for k, v in (preds.get("metadata") or {}).items()
               if k not in ("type", "model", "split", "batch_size", "batch_number")},
        },
        "inference_params": preds.get("inference_params") or {},
    }
    if per_type_metrics:
        output["per_type_metrics"] = per_type_metrics
    if retrieval_total_gold or retrieval_total_retrieved or retrieval_total_overlap:
        output["retrieval_metrics"] = {
            "total_gold": retrieval_total_gold,
            "total_retrieved": retrieval_total_retrieved,
            "total_overlap": retrieval_total_overlap,
            "precision": (
                round(retrieval_total_overlap / retrieval_total_retrieved, 4)
                if retrieval_total_retrieved
                else 0.0
            ),
            "recall": (
                round(retrieval_total_overlap / retrieval_total_gold, 4)
                if retrieval_total_gold
                else 0.0
            ),
        }
    return output


def build_output_filename(dataset: str, agent: str, llm: str, bn: int, bs: int, timestamp: str) -> str:
    """Compose the output filename using the required schema.

    Replaces unsafe characters in llm/agent with '-'.
    """
    def safe(s: str) -> str:
        s = s.replace("/", "-")
        s = s.replace(" ", "-")
        return s

    return f"{timestamp}_{dataset}_{safe(agent)}_{safe(llm)}_bn={bn}_bs={bs}.json"


def save_results(results: Dict[str, Any], outdir: str, filename: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return outpath
