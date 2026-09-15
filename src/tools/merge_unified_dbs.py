"""Merge multiple sharded unified_db JSON files into one big DB JSON.

The two_phase eval saves one unified_db JSON per run (see the --concat-all-db
path in ``src/eval_multihop.py``). When we shard a large dataset across many
GPUs and each shard builds its own unified DB, this script concatenates the
triplet lists from those shards so downstream Phase-2 runs can load a single
DB via ``--load-unified-db`` and retrieve against the full triplet pool.

Contract of the input JSON files (one per shard):

    {
      "metadata": {
        "dataset": "2wiki", "split": "dev", "num_examples": 786,
        "start_index": 0, "total_triplets": 205000,
        "use_contexts": "all", "contexts_are_split": true, ...
      },
      "triplets": [{"head": "...", "relation": "...", "tail": "..."}, ...]
    }

Output JSON has the same schema; ``metadata.merged_from`` records the shards.

Usage:
    python -m src.tools.merge_unified_dbs \\
        --inputs unified_db_2wiki_dev_n786_i0_all.json \\
                 unified_db_2wiki_dev_n786_i786_all.json \\
                 ... \\
        --output unified_db_2wiki_dev_MERGED_n12576_all.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger("merge_unified_dbs")


def load_shard(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "triplets" not in data:
        raise ValueError(f"{path}: missing 'triplets' key")
    return data


def merge(
    shards: list[dict[str, Any]],
    dedupe: bool = False,
) -> dict[str, Any]:
    if not shards:
        raise ValueError("merge() requires at least one shard")

    # Sanity-check that shards agree on the invariants Phase 2 will assume.
    ref = shards[0].get("metadata", {})
    for i, s in enumerate(shards[1:], start=1):
        m = s.get("metadata", {})
        for key in ("dataset", "split", "use_contexts", "contexts_are_split"):
            if key in ref and key in m and ref[key] != m[key]:
                raise ValueError(
                    f"shard {i} metadata mismatch on {key!r}: "
                    f"ref={ref[key]!r} shard={m[key]!r}"
                )

    all_triplets: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for s in shards:
        for t in s.get("triplets", []):
            key = (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
            if dedupe:
                if key in seen:
                    continue
                seen.add(key)
            all_triplets.append(t)

    shard_summary = [
        {
            "path": s.get("__source_path"),
            "num_examples": s.get("metadata", {}).get("num_examples"),
            "start_index": s.get("metadata", {}).get("start_index"),
            "total_triplets": s.get("metadata", {}).get("total_triplets"),
        }
        for s in shards
    ]

    total_examples = sum(
        (s.get("metadata", {}).get("num_examples") or 0) for s in shards
    )

    merged = {
        "metadata": {
            "dataset": ref.get("dataset"),
            "split": ref.get("split"),
            "use_contexts": ref.get("use_contexts"),
            "contexts_are_split": ref.get("contexts_are_split"),
            "num_examples": total_examples,
            "total_triplets": len(all_triplets),
            "dedupe": dedupe,
            "merged_from": shard_summary,
        },
        "triplets": all_triplets,
    }
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Shard unified_db JSON files to merge (space separated).",
    )
    ap.add_argument("--output", required=True, help="Merged unified_db JSON path.")
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact (head, relation, tail) duplicates. Off by default so "
        "triplet frequencies match what individual shards would have built.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    missing = [p for p in args.inputs if not os.path.isfile(p)]
    if missing:
        logger.error("missing shard files: %s", missing)
        sys.exit(1)

    shards = []
    for p in args.inputs:
        s = load_shard(p)
        s["__source_path"] = p
        shards.append(s)
        logger.info(
            "loaded %s (%d triplets, %d examples, start_index=%s)",
            p,
            len(s.get("triplets", [])),
            s.get("metadata", {}).get("num_examples"),
            s.get("metadata", {}).get("start_index"),
        )

    merged = merge(shards, dedupe=args.dedupe)
    logger.info(
        "merged: %d shards -> %d triplets across %d examples (dedupe=%s)",
        len(shards),
        len(merged["triplets"]),
        merged["metadata"]["num_examples"],
        args.dedupe,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
