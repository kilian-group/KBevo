"""Merge sharded two_phase generation JSONs and recompute EM/F1.

Each sharded eval writes a generation JSON like
``eval_<dataset>_<split>_<model>_n<count>_i<start>_..._tp.json`` under
``generations_<save_version>/`` and a summary metrics JSON under
``results_<save_version>/``. This merger:

    1. concats all shards' ``results`` dicts (the qid -> prediction map);
    2. drops duplicate qids (shards must not overlap, but if they do we
       take the first shard's prediction and warn);
    3. writes a single merged preds JSON with the union;
    4. calls ``src.eval.evaluate.evaluate_file`` to compute overall
       EM/F1/precision/recall over the merged predictions and prints them
       (also written next to the merged preds as a ``.metrics.json``).

Usage:
    python -m src.tools.merge_shard_results \\
        --shards generations_dir/eval_..._i0_..._tp.json \\
                 generations_dir/eval_..._i786_..._tp.json \\
                 ... \\
        --output merged_preds.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

# Late import: eval.evaluate has module-level side effects; keep it lazy so the
# merger stays runnable even when the eval env isn't fully importable.

logger = logging.getLogger("merge_shard_results")


def _stream_records(path: str, fields: tuple[str, ...] = ("pred", "gold_answer", "new_gold_answer", "mquake_split_type")):
    """Yield (qid, slim_record) pairs from a shard file without loading it whole.

    Shard files are enormous (GBs when Phase-2 traces are stored per turn).
    We only need fields required by ``src.eval.evaluate.evaluate_file`` to
    recompute EM/F1, so stream with ijson and drop everything else.
    """
    try:
        import ijson  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "streaming merger requires ijson; run `pip install ijson` "
            "in the mem env"
        ) from e
    with open(path, "rb") as f:
        # 'results.<qid>' → the per-example dict; ijson yields (key, value) pairs
        for qid, rec in ijson.kvitems(f, "results"):
            slim = {k: rec.get(k) for k in fields if k in rec}
            yield qid, slim


def _stream_metadata(path: str) -> dict[str, Any]:
    """Load only the top-level ``metadata`` block from a shard file."""
    import ijson  # type: ignore
    meta: dict[str, Any] = {}
    with open(path, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "" and event == "map_key" and value == "results":
                # done — we've passed metadata and inference_params
                break
            if prefix.startswith("metadata.") and event in ("string", "number", "boolean", "null"):
                key = prefix[len("metadata."):]
                # only take flat keys (skip nested provenance blobs)
                if "." not in key:
                    meta[key] = value
    return meta


def merge_shards(shard_paths: list[str]) -> dict[str, Any]:
    if not shard_paths:
        raise ValueError("no shards passed")

    merged_results: dict[str, Any] = {}
    duplicates = 0
    shard_metadata_list = []

    first_meta = None
    for path in shard_paths:
        meta = _stream_metadata(path)
        n_here = 0
        for qid, rec in _stream_records(path):
            if qid in merged_results:
                duplicates += 1
                continue
            merged_results[qid] = rec
            n_here += 1
        shard_metadata_list.append(
            {
                "path": path,
                "dataset": meta.get("dataset"),
                "split": meta.get("split"),
                "num_results": n_here,
                "start_index": meta.get("start_index"),
            }
        )
        if first_meta is None:
            first_meta = meta
        logger.info(
            "loaded %s: %d records (running total %d)",
            path,
            n_here,
            len(merged_results),
        )

    if duplicates:
        logger.warning(
            "encountered %d duplicate qids across shards; kept first-shard "
            "prediction for each",
            duplicates,
        )

    # Reuse the first shard's metadata for downstream evaluate_file — dataset
    # / split / setting need to be consistent across shards (they always are
    # when shards come from the same eval command with different --start-index).
    merged: dict[str, Any] = {
        "metadata": dict(first_meta or {}),
        "results": merged_results,
    }
    merged["metadata"]["merged_from"] = shard_metadata_list
    merged["metadata"]["merged_count"] = len(merged_results)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="Shard generation JSONs to merge (the eval_*.json files under "
        "generations_<save_version>/).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Merged preds JSON path.",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name for gold-answer lookup (defaults to the "
        "first shard's metadata).",
    )
    ap.add_argument("--setting", default=None, help="Override setting for eval.")
    ap.add_argument("--split", default=None, help="Override split for eval.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    missing = [p for p in args.shards if not os.path.isfile(p)]
    if missing:
        logger.error("missing shard files: %s", missing)
        sys.exit(1)

    merged = merge_shards(args.shards)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info("wrote merged preds: %s (%d records)", args.output, len(merged["results"]))

    # Recompute overall metrics on the union.
    from src.eval.evaluate import evaluate_file  # lazy import

    result = evaluate_file(
        preds_path=args.output,
        dataset=args.dataset,
        setting=args.setting,
        split=args.split,
    )
    metrics = result.get("metrics", {})
    print(json.dumps({"merged_metrics": metrics}, indent=2))

    metrics_path = args.output + ".metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"merged_metrics": metrics, "meta": result.get("meta")}, f, indent=2)
    logger.info("wrote metrics summary: %s", metrics_path)


if __name__ == "__main__":
    main()
