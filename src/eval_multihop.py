#!/usr/bin/env python3
"""CLI to run the agent over a dataset and save predictions under preds/.

- Instantiates LLM and Agent
- Loads HotpotQA via Hugging Face datasets
- Runs the agent over questions (optionally limited)
- Saves predictions with structure: preds/{type}/{dataset}_{setting}/{model}/{split}_seed={s}_bn={n}_bs={b}.json

The JSON format uses deduplicated metadata at the top level:
{
  "metadata": { model, split, batch_size, batch_number, type, seed, retrieval },
  "inference_params": { seed, temperature, max_tokens },
  "results": {
    "qid": { pred, gold_answer, gold_evidence, question, trace, evidence }
  }
}
"""

import argparse
import hashlib
import json
import os
import random
import sys
import logging
import gc
from typing import Dict, Any, List
from tqdm import tqdm
from datetime import datetime

# Fix OpenMP conflict when multiple libraries link to different OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from constants import REPO_ROOT
import warnings
import torch

from agent import get_agent, Agent
from llm import get_llm
from data import get_dataset, selected_rows_provenance
from data.hotpotqa import load_hotpotqa_rag_corpus
from data.musique import load_musique_rag_corpus, write_musique_rag_corpus_jsonl
from data.popqa import load_popqa_rag_corpus
from multi_lmlm.database.database_manager import build_databases_from_triplets_batch

# Import for TriviaQA sentence splitting
from nltk.tokenize import PunktSentenceTokenizer

def split_into_parts(items: list, num_parts: int) -> list[list]:
    """Split a list into num_parts parts where max and min lengths differ by at most 1."""
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if num_parts >= len(items):
        # Return each item as its own part (or empty parts if num_parts > len)
        return [[item] for item in items] + [[] for _ in range(num_parts - len(items))]

    base_size = len(items) // num_parts
    remainder = len(items) % num_parts

    parts = []
    start = 0
    for i in range(num_parts):
        part_size = base_size + (1 if i < remainder else 0)
        parts.append(items[start:start + part_size])
        start += part_size

    return parts


def split_trivia_qa_contexts(contexts: List[str], titles: List[str], min_chunk_length: int = 800) -> List[str]:
    """Split TriviaQA contexts into sentence groups with length >= min_chunk_length.

    For each context (wiki_context without title prefix), this function:
    1. Splits the context text into sentences using PunktSentenceTokenizer
    2. Groups sentences into chunks where each chunk has >= min_chunk_length chars
    3. Returns a list of contexts formatted as "title: chunk"

    Args:
        contexts: List of wiki_context strings (without title prefix)
        titles: List of titles parallel to contexts (from context_titles field)
        min_chunk_length: Minimum character length for each chunk (default: 800)

    Returns:
        List of split contexts formatted as "title: chunk"
    """
    tokenizer = PunktSentenceTokenizer()
    result_contexts = []

    for i, wiki_context in enumerate(contexts):
        # Get the title from the parallel list
        title = titles[i] if i < len(titles) else "Unknown"

        # Split wiki_context into sentences
        sentences = tokenizer.tokenize(wiki_context)

        # Group sentences into chunks of >= min_chunk_length
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence)

            # If we've reached the minimum length, create a new context chunk
            if current_length >= min_chunk_length:
                chunk_text = " ".join(current_chunk)
                result_contexts.append(f"{title}: {chunk_text}")
                current_chunk = []
                current_length = 0

        # Add any remaining sentences as a final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            result_contexts.append(f"{title}: {chunk_text}")

    return result_contexts

DEFAULT_FULLWIKI_CORPUS_PATH = os.environ.get("FULLWIKI_CORPUS_PATH")


def _load_training_config(model_path: str) -> dict:
    cfg = {}
    for fname in ("training_args.json", "trainer_state.json"):
        p = os.path.join(model_path, fname)
        if os.path.exists(p):
            with open(p) as f:
                cfg.update(json.load(f))
    return cfg


# Sampling params used during GRPO training (grpo_train.sh).
# Applied to TwoPhaseAgent when --use-train-params is set.
# max_model_len is set higher than training (4096) to handle multi-turn context growth in eval.
TRAINING_SAMPLING_PARAMS: dict = {
    "temperature": 1.0,
    "top_p": 0.95,
    "vllm_top_k": 4,
    "repetition_penalty": 1.0,
    "max_completion_length": 1024,
    "max_model_len": 8192,
}



from eval.evaluate import (
    evaluate_file,
    build_output_filename,
    save_results,
)

def build_query(question: str) -> str:
    """Instruction to ensure the Agent emits a FINAL_ANSWER the parser recognizes."""
    instruction = "Provide only the final answer prefixed by 'FINAL_ANSWER:' with no extra text."
    return f"{instruction}\n{question}"


def _infer_rag_scope(rag_corpus_path: str) -> str:
    lower_name = os.path.basename(rag_corpus_path).lower()
    if "fullwiki" in lower_name:
        return "fullwiki"
    if "distractor" in lower_name:
        return "distractor"
    return "custom"


def _normalize_title(title: Any) -> str:
    return str(title or "").strip().lower()


def _extract_retrieved_title(doc: Any) -> str:
    if isinstance(doc, dict):
        return _normalize_title(doc.get("title", ""))
    return ""


def _compute_retrieval_stats(
    evidence_docs: List[Any],
    supporting_facts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    supporting_titles = {
        _normalize_title(item.get("title", ""))
        for item in (supporting_facts or [])
        if isinstance(item, dict)
    }
    supporting_titles = {t for t in supporting_titles if t}
    retrieved_titles = {
        _extract_retrieved_title(doc) for doc in (evidence_docs or [])
    }
    retrieved_titles = {t for t in retrieved_titles if t}

    overlap = supporting_titles.intersection(retrieved_titles)
    gold_total = len(supporting_titles)
    retrieved_total = len(retrieved_titles)
    overlap_count = len(overlap)

    return {
        "gold_total": gold_total,
        "retrieved_total": retrieved_total,
        "overlap": overlap_count,
        "precision": overlap_count / retrieved_total if retrieved_total else 0.0,
        "recall": overlap_count / gold_total if gold_total else 0.0,
    }


def process_single_batch(
    args: argparse.Namespace,
    batch_number: int,
    total_examples: int,
    full_dataset,
    agent: Agent,
    existing_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Process a single batch and return results dict."""
    logger = logging.getLogger("run_agent")

    # Calculate batch indices starting from start_index
    start_idx = args.start_index + (batch_number - 1) * args.batch_size
    end_idx = min(start_idx + args.batch_size, total_examples)

    if start_idx >= total_examples:
        return {}

    # Select batch slice
    ds = full_dataset.select(range(start_idx, end_idx))

    # Check if this batch is already complete (for resume)
    if args.resume and len(ds) > 0:
        first_qid = str(ds[0].get("id") or ds[0].get("_id"))
        if first_qid in existing_results:
            logger.info(f"Skipping batch {batch_number} (already processed)")
            return {}

    logger.info(
        "Processing batch %d: indices [%d, %d) => %d examples",
        batch_number,
        start_idx,
        end_idx,
        len(ds),
    )

    # Run predictions
    results: Dict[str, Dict[str, Any]] = {}
    batch_size_actual = len(ds)

    # Collect queries and info for batch
    queries = []
    examples_metadata = []

    for ex in ds:
        qid = ex.get("id") or ex.get("_id") or ex.get("case_id")
        question = ex["question"]
        contexts = ex.get("contexts") or []

        # Handle context selection for two_phase method
        if args.method == "two_phase":
            if args.use_contexts == "golden":
                # two_phase was trained on golden_contexts (2 supporting paragraphs), not all 10
                # distractor paragraphs; use golden_contexts to match training conditions.
                print("using golden context!!")
                contexts = ex.get("golden_contexts")
            elif args.use_contexts == "all":
                # Use all contexts and split into parts
                original_len = len(contexts)

                # TriviaQA uses custom sentence-based splitting to avoid token limits
                if args.dataset.lower() in {"trivia_qa", "triviaqa", "popqa"}:
                    # Get context_titles from the example (parallel to contexts)
                    context_titles = ex.get("context_titles", [])
                    # Split articles into sentence chunks, then wrap each chunk as its own part
                    chunks = split_trivia_qa_contexts(contexts, context_titles, min_chunk_length=600)
                    contexts = [[chunk] for chunk in chunks]  # Each chunk becomes its own part
                    if len(queries) == 0:  # Log only for first example
                        logger.info(f"Split {original_len} TriviaQA contexts into {len(chunks)} parts (1 chunk per part)")
                else:
                    # TODO: Check if 5 parts is appropriate for musique, hotpot, and 2wiki dataset sizes
                    contexts = split_into_parts(contexts, num_parts=5)
                    if len(queries) == 0:  # Log only for first example
                        logger.info(f"Split {original_len} contexts into {len(contexts)} parts for first example")

        queries.append(question)
        
        metadata_entry = {
            "qid": qid,
            "question": question,
            "answers": ex["answers"],
            "supporting_facts": ex.get("supporting_facts"),
            "contexts": contexts,
        }
        for provenance_key in (
            "source_index",
            "eval_position",
            "confiqa_setting",
            "is_counterfactual",
        ):
            if provenance_key in ex:
                metadata_entry[provenance_key] = ex[provenance_key]
        # For MQuAKE, also capture new_answers and split type for knowledge editing evaluation
        if "new_answers" in ex:
            metadata_entry["new_answers"] = ex["new_answers"]
        if "mquake_split_type" in ex:
            metadata_entry["mquake_split_type"] = ex["mquake_split_type"]
        examples_metadata.append(metadata_entry)

    logger.info(f"Processing batch of {len(queries)} queries")

    if args.method in ("rag", "icl"):
        agent.reset(contexts)  # type: ignore

    if args.method == "direct":
        answers = []
        traces = []
        for query in queries:
            # Ensure each question starts with a fresh trace
            agent.trace = []
            answer_list, trace_list = agent.run(
                [query],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            answers.append(answer_list[0] if answer_list else None)
            traces.append(trace_list[0] if trace_list else None)
    elif args.method == "two_phase":
        contexts_list = [meta["contexts"] for meta in examples_metadata]
        answers, traces = agent.run(
            queries,
            contexts=contexts_list,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        answers, traces = agent.run(
            queries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    # Format results
    for idx, metadata in tqdm(enumerate(examples_metadata), total=len(examples_metadata), desc="Processing queries"):

        answer = answers[idx] if idx < len(answers) else None
        trace = traces[idx] if idx < len(traces) else None

        # Extract evidence docs for RAG if needed
        evidence_docs = []
        if args.method == "rag":
            evidence_docs = getattr(agent, "_evidence_docs", [])
        logger.debug("Answer: %s", answer)
        logger.debug("Trace: %s", trace)

        # Fallback to last step text if FINAL_ANSWER not parsed
        if answer is None and trace:
            answer = (trace[-1].answer or "").strip()
        if answer is None:
            answer = ""

        # Serialize trace for JSON output
        serialized_trace = [
            {
                "prompt": step.prompt,
                "answer": step.answer,
                "action": step.action,
                "error": step.error,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
                "tool_result": step.tool_result,
                "golden_triplets" : step.golden_triplets,
            }
            for step in (trace or [])
        ]

        results[str(metadata["qid"])] = {
            "pred": answer,
            "gold_answer": metadata["answers"],
            "gold_evidence": metadata["supporting_facts"],
            "question": metadata["question"],
            "trace": serialized_trace,
        }
        for provenance_key in (
            "source_index",
            "eval_position",
            "confiqa_setting",
            "is_counterfactual",
        ):
            if provenance_key in metadata:
                results[str(metadata["qid"])][provenance_key] = metadata[provenance_key]
        # For MQuAKE, also save new_gold_answer and split type for knowledge editing evaluation
        if "new_answers" in metadata:
            results[str(metadata["qid"])]["new_gold_answer"] = metadata["new_answers"]
        if "mquake_split_type" in metadata:
            results[str(metadata["qid"])]["mquake_split_type"] = metadata["mquake_split_type"]
        if args.method == "lmlm":
            lookup_logs = getattr(agent, "_lookup_logs", [])
            if idx < len(lookup_logs):
                results[str(metadata["qid"])]["lookup_logs"] = lookup_logs[idx]
        if args.method == "two_phase":
            lookup_logs = getattr(agent, "_lookup_logs", [])
            if idx < len(lookup_logs):
                results[str(metadata["qid"])]["lookup_logs"] = lookup_logs[idx]
            phase1_info = getattr(agent, "_phase1_info", [])
            if idx < len(phase1_info):
                results[str(metadata["qid"])]["phase1"] = phase1_info[idx]
        if args.method == "rag":
            results[str(qid)]["retrieval"] = _compute_retrieval_stats(
                evidence_docs=evidence_docs,
                supporting_facts=ex.get("supporting_facts") or [],
            )
        if args.method == "rag" and args.debug_evidence:
            results[str(metadata["qid"])]["evidence"] = evidence_docs

    logger.info("Generated %d predictions for batch %d", len(results), batch_number)

    # Force garbage collection and clear GPU cache to prevent memory fragmentation
    gc.collect()
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results


def _fit_filename(name: str, max_bytes: int = 255) -> str:
    """Shorten a leaf filename that would exceed the filesystem's 255-byte cap.

    Long model tags push results_*.json past the limit (hotpotqa lands on 256
    bytes and dies with "Errno 36 File name too long", while musique squeaks
    through at 255). The model tag is also the parent directory name, so
    dropping the middle of it loses nothing; a hash of the full name is spliced
    in so two different runs can never collide. No-op for names that already
    fit, which keeps every existing path untouched.
    """
    if len(name.encode()) <= max_bytes:
        return name
    stem, _, ext = name.rpartition(".")
    ext = f".{ext}" if stem else ""
    stem = stem or name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    budget = max_bytes - len(ext.encode()) - len(digest) - 2  # 2 for the ".." join
    head = budget * 2 // 3
    return f"{stem[:head]}.{digest}.{stem[-(budget - head):]}{ext}"


def save_results_to_file(
    all_results: Dict[str, Dict[str, Any]],
    save_path: str,
    args: argparse.Namespace,
) -> None:
    """Save results to JSON file."""
    logger = logging.getLogger("run_agent")

    # Build final output with metadata
    # Use unified_db_path if it was created, otherwise use args.database_path
    database_path_value = getattr(args, 'unified_db_path', None) or args.database_path

    output = {
        "metadata": {
            "model-path": args.model_path,
            "database-path": database_path_value,
            "model": args.model,
            "dataset": args.dataset,
            "setting": args.confiqa_setting if args.dataset == "confiqa" else args.setting,
            "dataset_source": args.dataset_source,
            "dataset_provenance": getattr(args, "_dataset_provenance", None),
            "split": args.split,
            "batch_size": args.batch_size,
            "total_examples": len(all_results),
            "type": args.method,
            "seed": args.seed if args.seed is not None else None,
        },
        "inference_params": {
            "seed": args.seed,
            **getattr(args, "_effective_sampling", {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }),
        },
        "results": all_results,
    }
    # Add two_phase-specific params
    if args.method == "two_phase":
        output["metadata"]["two_phase_params"] = {
            "phase1_prompt_type": args.phase1_prompt_type,
            "top_k": args.top_k,
            "similarity_threshold": args.similarity_threshold,
            "concat_all_db": args.concat_all_db,
            "use_contexts": args.use_contexts,
        }
    # Add retrieval metadata for RAG
    if args.method == "rag":
        output["metadata"]["retrieval"] = {
            "backend": args.retrieval,
            "scope": args.setting,
            "k": args.rag_k,
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info("Saved %d predictions to %s", len(all_results), save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent over a dataset and save predictions.")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "2wiki", "synthworlds", "trivia_qa", "popqa", "confiqa", "mquake", "mquake-remastered"], help="Dataset name")
    parser.add_argument(
        "--dataset-source",
        default="auto",
        choices=["auto", "hf", "remote", "local"],
        help="Use pinned public data by default, or an explicitly configured local override.",
    )
    parser.add_argument(
        "--setting",
        default="distractor",
        choices=["distractor", "fullwiki", "long_tail", "qa-sm", "qa-rm", "rc.wikipedia"],
        help="Dataset setting",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "validation", "test", "eval-edit", "eval-edit-new", "eval-original"],
        help="Dataset split",
    )
    parser.add_argument(
        "--method",
        default="icl",
        choices=["db", "rag", "icl", "lmlm", "direct", "two_phase"],
        help="Agent method label (for output path)",
    )
    parser.add_argument(
        "--phase1-prompt-type",
        default="sft",
        choices=["sft", "with_question"],
        help="Phase 1 prompt template for two_phase method",
    )
    parser.add_argument(
        "--use-contexts",
        default="golden",
        choices=["golden", "all"],
        help="Use golden contexts or all contexts (two_phase only). If 'all', contexts are split into parts.",
    )
    parser.add_argument(
        "--concat-all-db",
        action="store_true",
        help="Build a single unified database from all examples (two_phase only)",
    )
    parser.add_argument(
        "--load-unified-db",
        default=None,
        help=(
            "Path to a prebuilt unified_db JSON (from a previous --concat-all-db run or "
            "src/tools/merge_unified_dbs.py). Skips Phase 1 entirely and builds the "
            "DatabaseManager from the saved triplets. Requires --method=two_phase; "
            "implies --concat-all-db semantics for Phase 2."
        ),
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help=(
            "Build and save the unified database, then exit before Phase 2. Used to "
            "produce a DB JSON that later runs can load via --load-unified-db. "
            "Requires --concat-all-db --method=two_phase."
        ),
    )
    parser.add_argument(
        "--confiqa-setting",
        default="orig",
        choices=["orig", "cf", "cf_100", "cf_500"],
        help="ConFiQA setting: 'orig' for original, 'cf' for counterfactual, 'cf_100' for CF on 100 examples, 'cf_500' for CF on 500 examples (only used with --dataset confiqa)",
    )
    parser.add_argument("--model-path", default=None, help="Local model path")
    parser.add_argument(
        "--database-path",
        default=None,
        help="Path to database of (entity, relation, value) triplets",
    )
    parser.add_argument(
        "--adaptive-k",
        default=False,
        help="Whether to use adaptive k for lmlm retreival",
        action="store_true"
    )
    parser.add_argument(
        "--top-k",
        default=4,
        type=int,
        help="Maximum number of results to retrieve from database",
    )
    parser.add_argument(
        "--similarity-threshold",
        default=0.9,
        type=float,
        help="cosine similarity threshold for lmlm retrieval",
    )
    parser.add_argument(
        "--return-triplets",
        default=False,
        help="Whether to return entire triplets (as opposed to only values)",
        action="store_true"
    )
    # RAG-related flags
    parser.add_argument(
        "--retrieval", default="bm25", choices=["bm25"], help="Retrieval backend for --method rag"
    )
    parser.add_argument("--rag-k", type=int, default=4, help="Top-k documents to retrieve")
    parser.add_argument(
        "--search-r1-retrieval-url",
        default="http://127.0.0.1:8000/retrieve",
        help="Search-R1 retrieval service endpoint",
    )
    parser.add_argument(
        "--search-r1-retrieval-k", type=int, default=3, help="Search-R1 documents per query"
    )
    parser.add_argument(
        "--search-r1-protocol",
        choices=["official", "hermes"],
        default="official",
        help=(
            "Which tool-calling protocol the checkpoint was trained with. "
            "'official' = Qwen2.5 EM-PPO release (<search>/<information>); "
            "'hermes' = custom Qwen3 icl3hop-trained checkpoints "
            "(<tool_call>/<tool_response>). Mixing produces plausible-but-invalid "
            "rollouts, tanking scores by 20-30 EM points."
        ),
    )
    parser.add_argument(
        "--search-r1-prompt-variant",
        choices=["default", "thinkingtag", "icl3hop"],
        default="icl3hop",
        help=(
            "Hermes-protocol only: which USER_PREFIX matches the training run. "
            "'default' = <think> tag (early runs / 4B@3e-6). "
            "'thinkingtag' = <thinking> tag (Run D). "
            "'icl3hop' = <thinking> + 3-hop ICL example (Run C, our step500 ckpts)."
        ),
    )
    parser.add_argument(
        "--search-r1-enable-thinking",
        action="store_true",
        help=(
            "Hermes-protocol only: pass enable_thinking=True to Qwen3's chat "
            "template. Default False mirrors how the icl3hop checkpoints were "
            "trained."
        ),
    )
    # Accepted-but-ignored deprecated flags. Kept so in-flight jobs 859612-617
    # can requeue without argparse rejecting them. Safe to delete once those
    # runs (and any resubmissions) have finished.
    for _dep in (
        "--search-r1-doc-format",
        "--search-r1-stop-variants",
    ):
        parser.add_argument(_dep, help=argparse.SUPPRESS)
    parser.add_argument(
        "--search-r1-splice-prefix-newlines", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--search-r1-max-tool-response-length",
        type=int,
        default=4096,
        help=(
            "Maximum Search-R1 tool response length in characters. Upstream "
            "does not truncate at all; 512 cut three retrieved documents down "
            "to a fragment and starved the model of the evidence it searched "
            "for. 4096 fits a typical top-3 BM25 response intact."
        ),
    )
    parser.add_argument(
        "--search-r1-tool-response-truncate-side",
        choices=["left", "right", "middle"],
        default="left",
    )
    parser.add_argument(
        "--search-r1-retrieval-timeout", type=float, default=30.0
    )
    parser.add_argument(
        "--search-r1-retrieval-workers", type=int, default=32
    )
    parser.add_argument(
        "--search-r1-max-response-length",
        type=int,
        default=2048,
        help="Total Search-R1 trajectory token budget, including tool responses",
    )
    parser.add_argument(
        "--search-r1-max-model-len", type=int, default=3072
    )
    parser.add_argument(
        "--search-r1-dtype",
        default="bfloat16",
        help=(
            "vLLM dtype for search_r1. The released checkpoints declare "
            "torch_dtype=float32 (verl saves fp32 FSDP master weights) even "
            "though training and rollout ran in bf16, so bf16 both matches how "
            "the weights were produced and halves memory. Use 'auto' to follow "
            "config.json instead."
        ),
    )
    # Shared by search_r1 and lmlm, which want different defaults (0.95 vs the
    # historical greedy 1.0). Left as None here and resolved per-method after
    # parse_args() so neither method's default behaviour changes.
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling probability (default: 0.95 for search_r1, 1.0 for lmlm)")
    parser.add_argument(
        "--sampling-top-k", type=int, default=-1, help="Sampling top-k; -1 disables it"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--debug-evidence",
        action="store_true",
        help="Include retrieved evidence in saved preds for debugging",
    )
    parser.add_argument(
        "--rag-corpus-path",
        default=None,
        help=(
            "Optional path to a HotpotQA/MuSiQue JSON or JSONL file to build a global RAG corpus. "
            "For MuSiQue, you can pass 'hf:<split>' (e.g., hf:train) to build and cache a JSONL."
        ),
    )

    # NOTE: --top-p is declared once above (shared with search_r1); for lmlm it
    # defaults to 1.0. two_phase resolves top_p from --use-train-params instead.
    parser.add_argument("--vllm-top-k", type=int, default=0,
                        help="lmlm only: vLLM sampling top_k, 0 = disabled (distinct from retrieval --top-k)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max model length for two_phase (default 8192 > training 4096 to handle multi-turn context growth)")

    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (two_phase greedy default; overridden by --use-train-params)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max completion tokens (two_phase greedy default; overridden by --use-train-params)")
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Max reasoning steps for the Agent"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index to start from (0-based). Useful for parallelization.",
    )
    parser.add_argument(
        "--total-count",
        type=int,
        default=1000,
        help="Total number of examples to process from start-index. Default is 1000.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save results every N batches. None saves once at the end",
    )
    parser.add_argument(
        "--use-inverses",
        default=False,
        help="Whether to allow inverse lookups, of the form (value, relationship)",
        action="store_true"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file, skipping already processed examples.",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Base output directory (defaults to <repo>/preds)"
    )
    parser.add_argument(
        "--save-version", default=None, help="Save version (defaults to "")"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the predictions",
    )
    parser.add_argument(
        "--use-train-params",
        action="store_true",
        help=(
            "For two_phase: read temperature, top_p, top_k, repetition_penalty, "
            "max_completion_length, and vllm_max_model_length from the checkpoint's "
            "training_args.json instead of CLI defaults."
        ),
    )
    args = parser.parse_args()

    # --top-p is shared by search_r1 and lmlm but their historical defaults
    # differ, so it is declared with default=None and resolved here. lmlm's 1.0
    # (with --vllm-top-k 0) reproduces the original greedy decoding.
    if args.top_p is None:
        args.top_p = 1.0 if args.method == "lmlm" else 0.95

    # Validate use-contexts flag
    if args.use_contexts == "all" and args.method != "two_phase":
        raise ValueError("--use-contexts=all is only supported for --method=two_phase")

    # SynthWorlds does not have a 'contexts' field, only 'golden_contexts'
    if args.dataset.lower() in {"synthworlds", "synth"} and args.use_contexts == "all":
        raise ValueError("--use-contexts=all is not supported for SynthWorlds dataset (only 'golden' contexts are available)")

    if args.concat_all_db and args.method != "two_phase":
        raise ValueError("--concat-all-db is only supported for --method=two_phase")

    # --load-unified-db short-circuits Phase 1; it is a Phase-2-only variant of
    # --concat-all-db, so gate on the same method and treat concat_all_db as implied.
    if args.load_unified_db:
        if args.method != "two_phase":
            raise ValueError("--load-unified-db is only supported for --method=two_phase")
        if args.phase1_only:
            raise ValueError("--load-unified-db and --phase1-only are mutually exclusive")
        if not os.path.exists(args.load_unified_db):
            raise FileNotFoundError(f"--load-unified-db path does not exist: {args.load_unified_db}")
        args.concat_all_db = True  # unified-DB code paths key off this

    if args.phase1_only and not (args.concat_all_db and args.method == "two_phase"):
        raise ValueError("--phase1-only requires --concat-all-db --method=two_phase")

    if args.method == "search_r1":
        if not args.model_path:
            raise ValueError("--model-path is required for --method=search_r1")
        if args.search_r1_retrieval_k <= 0:
            raise ValueError("--search-r1-retrieval-k must be positive")
        if args.search_r1_max_tool_response_length <= 0:
            raise ValueError("--search-r1-max-tool-response-length must be positive")
        if args.search_r1_retrieval_workers <= 0:
            raise ValueError("--search-r1-retrieval-workers must be positive")
        if args.search_r1_max_response_length <= 0 or args.search_r1_max_model_len <= 0:
            raise ValueError("Search-R1 response and model lengths must be positive")
        args.max_tokens = args.search_r1_max_response_length

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [run_agent] %(message)s",
    )
    logger = logging.getLogger("run_agent")

    # For two_phase: resolve sampling params.
    # --use-train-params → use TRAINING_SAMPLING_PARAMS (grpo_train.sh values).
    # default           → greedy eval (T=0, top_p=1, top_k=-1).
    _extra_agent_kwargs: dict = {}
    if args.method == "two_phase":
        if args.use_train_params:
            _extra_agent_kwargs = dict(TRAINING_SAMPLING_PARAMS)
            logger.info(
                "[two_phase] using training sampling params: T=%.3f  top_p=%.3f  vllm_top_k=%d  "
                "rep_penalty=%.3f  max_tokens=%d  max_model_len=%d",
                _extra_agent_kwargs["temperature"], _extra_agent_kwargs["top_p"],
                _extra_agent_kwargs["vllm_top_k"], _extra_agent_kwargs["repetition_penalty"],
                _extra_agent_kwargs["max_completion_length"], _extra_agent_kwargs["max_model_len"],
            )
        else:
            _extra_agent_kwargs = {
                "temperature": 0.0,
                "top_p": 1.0,
                "vllm_top_k": -1,
                "repetition_penalty": 1.0,
                "max_completion_length": args.max_tokens,
                "max_model_len": args.max_model_len,
            }
            logger.info(
                "[two_phase] using greedy eval params: T=0.0  top_p=1.0  vllm_top_k=-1  "
                "max_tokens=%d  max_model_len=%d",
                _extra_agent_kwargs["max_completion_length"], _extra_agent_kwargs["max_model_len"],
            )
        # Sync args.temperature / args.max_tokens so agent.run() call also uses these values
        args.temperature = _extra_agent_kwargs["temperature"]
        args.max_tokens  = _extra_agent_kwargs["max_completion_length"]

    # Store effective sampling params on args for metadata saving
    args._effective_sampling = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        **({k: _extra_agent_kwargs[k] for k in ("top_p", "vllm_top_k", "repetition_penalty", "max_model_len")}
           if args.method == "two_phase" else {}),
        **({"top_p": args.top_p, "vllm_top_k": args.vllm_top_k, "max_model_len": args.max_model_len}
           if args.method == "lmlm" else {}),
        **({
            "top_p": args.top_p,
            "sampling_top_k": args.sampling_top_k,
            "max_model_len": args.search_r1_max_model_len,
            "dtype": args.search_r1_dtype,
        } if args.method == "search_r1" else {}),
        "use_train_params": getattr(args, "use_train_params", False),
    }

    if (
        args.method == "rag"
        and args.dataset == "hotpotqa"
        and args.setting == "fullwiki"
        and not args.rag_corpus_path
    ):
        args.rag_corpus_path = DEFAULT_FULLWIKI_CORPUS_PATH
        if not args.rag_corpus_path:
            raise SystemExit(
                "HotpotQA RAG fullwiki mode requires an explicit --rag-corpus-path "
                "or FULLWIKI_CORPUS_PATH env var pointing at a fullwiki context JSON "
                "(e.g. hotpot_dev_fullwiki_v1.json). Canonical KBevo two_phase "
                "evaluation does not need this."
            )
    if args.method == "rag" and args.dataset == "musique" and not args.rag_corpus_path:
        args.rag_corpus_path = f"hf:{args.split}"
        logger.info(
            "MuSiQue RAG requires a global corpus; defaulting to %s",
            args.rag_corpus_path,
        )

    rag_corpus = None
    rag_corpus_path = args.rag_corpus_path
    rag_scope = None
    if (
        args.method == "rag"
        and args.dataset == "musique"
        and isinstance(rag_corpus_path, str)
        and rag_corpus_path.startswith("hf:")
    ):
        hf_split = rag_corpus_path.split(":", 1)[1].strip() or "train"
        cache_dir = os.path.join(REPO_ROOT, "preds", "rag_corpus")
        rag_corpus_path = os.path.join(cache_dir, f"musique_{hf_split}.jsonl")
        logger.info(
            "Building MuSiQue RAG corpus from HF split=%s and saving to %s",
            hf_split,
            rag_corpus_path,
        )
        count = write_musique_rag_corpus_jsonl(
            path=rag_corpus_path,
            split=hf_split,
            limit=None,
            seed=args.seed,
        )
        logger.info("Wrote %d unique RAG paragraphs to %s", count, rag_corpus_path)
        rag_scope = f"hf_{hf_split}"

    if args.method == "rag" and rag_corpus_path:
        logger.info("Loading RAG corpus from %s", rag_corpus_path)
        if args.dataset == "hotpotqa":
            rag_corpus = load_hotpotqa_rag_corpus(rag_corpus_path)
            logger.info(f"Loaded {len(rag_corpus)} unique RAG paragraphs from hotpotqa")
        elif args.dataset == "musique":
            rag_corpus = load_musique_rag_corpus(rag_corpus_path)
            logger.info(f"Loaded {len(rag_corpus)} unique RAG paragraphs from musique")
        elif args.dataset == "popqa":
            rag_corpus = load_popqa_rag_corpus(rag_corpus_path)
            logger.info(f"Loaded {len(rag_corpus)} unique RAG articles from PopQA")
        else:
            rag_corpus = []
        
        # if rag corpus loading fails
        if not rag_corpus:
            logger.warning(
                "RAG corpus is empty after loading %s (check format and content).",
                rag_corpus_path,
            )
            raise RuntimeError(
                "RAG requires a non-empty global corpus; "
                "please verify --rag-corpus-path or HF cache."
            )

    if rag_scope is None:
        if args.method == "rag" and rag_corpus_path:
            rag_scope = _infer_rag_scope(rag_corpus_path)
        else:
            rag_scope = args.setting
    args.rag_scope = rag_scope

    print("split is :", args.split)

    # Load full dataset once (with seed for deterministic shuffling)
    # BUG: either use start_index or sub_split, not both
    # if start_index is used, sub_split should be None
    # if args.start_index is not None:
    #     sub_split = None
    #     if args.sub_split is not None:
    #         warnings.warn("start_index is used, sub_split will be ignored during dataset loading")
    # else:
    #     sub_split = args.sub_split

    # For ConFiQA, use confiqa_setting instead of setting
    dataset_setting = args.confiqa_setting if args.dataset == "confiqa" else args.setting
    if args.dataset == "confiqa":
        logger.info(f"DEBUG: Loading ConFiQA with setting='{dataset_setting}' (args.confiqa_setting='{args.confiqa_setting}')")
    full_dataset = get_dataset(
        name=args.dataset,
        setting=dataset_setting,
        split=args.split,
        source=args.dataset_source,
        seed=args.seed,
    )
    total_dataset_size = len(full_dataset)

    print(f"examples in dataset: {full_dataset[0]}")

    # Validate start_index
    if args.start_index >= total_dataset_size:
        logger.warning(f"Start index {args.start_index} is at or beyond dataset size {total_dataset_size}")
        return

    # Calculate how many examples to process (total_count is NUMBER of examples from start_index)
    examples_to_process = min(args.total_count, total_dataset_size - args.start_index)

    # TODO: how to make the training and eval use the same split function (e.g. create_train_val_splits)?
    # Calculate the exclusive end index
    end_index = args.start_index + examples_to_process

    selected_dataset = full_dataset.select(range(args.start_index, end_index))
    counterfactual_count = None
    if "is_counterfactual" in selected_dataset.column_names:
        counterfactual_count = sum(
            bool(value) for value in selected_dataset["is_counterfactual"]
        )
    id_column = next(
        (
            column
            for column in ("id", "_id", "case_id")
            if column in selected_dataset.column_names
        ),
        None,
    )
    selected_ids = (
        selected_dataset[id_column]
        if id_column is not None
        else list(range(args.start_index, end_index))
    )
    args._dataset_provenance = selected_rows_provenance(
        args.dataset,
        selected_ids,
        seed=args.seed,
        setting=dataset_setting,
        counterfactual_count=counterfactual_count,
    )

    print(f"Evaluating {examples_to_process} / {total_dataset_size} examples (index {args.start_index} to {end_index})")
    logger.info(f"Dataset size: {total_dataset_size}, Processing {examples_to_process} examples from index {args.start_index} to {end_index}")

    # Prepare output location
    base_output_dir = args.output_dir or os.path.join(REPO_ROOT, "preds")
    if args.method in ('lmlm', 'two_phase', 'search_r1'):
        # Strip trailing slashes to ensure consistent path parsing
        model_path_clean = args.model_path.rstrip('/')
        model_name = model_path_clean.split('/')[-1] if "checkpoint" not in model_path_clean else model_path_clean.split('/')[-2]+"-ckpt"+model_path_clean.split('/')[-1].split("checkpoint-")[-1]
        output_dir = os.path.join(base_output_dir, args.method, args.dataset, model_name)
        use_inv_str = "_inv" if args.use_inverses else ""
        # Encode all settings that vary across runs to prevent file overwrite
        settings_str = ""
        if args.method == "two_phase":
            settings_str += f"_ctx{args.use_contexts}"
            if args.concat_all_db:
                settings_str += "_cdb"
        if args.method != "search_r1":
            settings_str += f"_k{args.top_k}"
        if getattr(args, "use_train_params", False):
            settings_str += "_tp"
        if args.dataset == "synthworlds":
            settings_str += f"_setting{args.setting}"
        if args.dataset == "confiqa":
            settings_str += f"_setting{args.confiqa_setting}"

        save_postfix = f"{args.dataset}_{args.split}_{model_name}_n{examples_to_process}_i{args.start_index}{use_inv_str}{settings_str}.json"
        save_path = os.path.join(output_dir, f"generations{args.save_version}", _fit_filename(f"eval_{save_postfix}"))
        save_results_path = os.path.join(output_dir, f"results{args.save_version}", _fit_filename(f"results_{save_postfix}"))
    else:
        model_name = args.model or "unknown-model"
        save_path = os.path.join(base_output_dir, args.method, f"{args.dataset}_{args.setting}", model_name, "generations")
        save_results_path = os.path.join(base_output_dir, args.method, f"{args.dataset}_{args.setting}", model_name, "results")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_results_path), exist_ok=True)

    # Load existing results if resuming
    existing_results = {}
    if args.resume and os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                existing_data = json.load(f)
                existing_results = existing_data.get("results", {})
                logger.info(f"Resuming from {save_path} with {len(existing_results)} existing results")
        except Exception as e:
            logger.warning(f"Failed to load existing results from {save_path}: {e}")

    # Check if we already have all the results we need (and not resuming)
    if args.resume and os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                existing_data = json.load(f)

            if (len(existing_data["results"]) >= examples_to_process and
                existing_data["metadata"]["model-path"] == args.model_path):
                logger.info(f"Generations already complete at {save_path} ({len(existing_data['results'])} results). Evaluating...")
                if args.eval:
                    # Evaluate
                    results = evaluate_file(
                        save_path,
                        dataset=args.dataset,
                        setting=args.setting,
                        split=args.split,
                        source='hf',
                    )
                    logger.info(f"Evaluation results: {json.dumps(results, indent=2)}")

                    outpath = save_results(results, "./", save_results_path)
                    logger.info(f"Evaluation results saved to: {outpath}")
                return
        except Exception as e:
            logger.warning(f"Failed to read existing results from {save_path}: {e}")

    # Calculate number of batches needed
    batches_to_process = (examples_to_process + args.batch_size - 1) // args.batch_size

    logger.info(
        "Processing %d examples in %d batches (starting from index %d): dataset=%s setting=%s split=%s method=%s model=%s batch_size=%d",
        examples_to_process,
        batches_to_process,
        args.start_index,
        args.dataset,
        args.setting,
        args.split,
        args.method,
        args.model,
        args.batch_size,
    )

    llm = None
    if args.method not in ("lmlm", "two_phase", "search_r1"):
        llm = get_llm(model_name=args.model)


    # Build agent_kwargs dictionary
    agent_kwargs = {
        "llm": llm,
        "model": args.model,
        "dataset": args.dataset,
        "setting": args.setting,
        "retrieval": args.retrieval,
        "rag_k": args.rag_k,
        "max_steps": args.max_steps,
        "model_path": args.model_path,
        "database_path": args.database_path,
        "return_triplets" : args.return_triplets,
        "use_inverses" : args.use_inverses,
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "phase1_prompt_type": args.phase1_prompt_type,
        "concat_all_db": args.concat_all_db if args.method == "two_phase" else False,
        "contexts_are_split": args.use_contexts == "all" if args.method == "two_phase" else False,
    }

    if args.method == "search_r1":
        agent_kwargs.update({
            "protocol": args.search_r1_protocol,
            # prompt_variant / enable_thinking are consumed only by the
            # SearchR1HermesAgent constructor; SearchR1Agent's **_ swallows them.
            "prompt_variant": args.search_r1_prompt_variant,
            "enable_thinking": args.search_r1_enable_thinking,
            "retrieval_url": args.search_r1_retrieval_url,
            "retrieval_top_k": args.search_r1_retrieval_k,
            "retrieval_timeout": args.search_r1_retrieval_timeout,
            "retrieval_workers": args.search_r1_retrieval_workers,
            "top_p": args.top_p,
            "sampling_top_k": args.sampling_top_k,
            "max_model_len": args.search_r1_max_model_len,
            # "auto" means "let vLLM read config.json"; the agent treats None
            # as that, so translate here rather than passing the string through.
            "dtype": None if args.search_r1_dtype == "auto" else args.search_r1_dtype,
            "max_tool_response_length": args.search_r1_max_tool_response_length,
            "tool_response_truncate_side": args.search_r1_tool_response_truncate_side,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
        })

    # Add RAG corpus if available (for fullwiki setting)
    if args.method == "rag" and rag_corpus:
        agent_kwargs["corpus"] = rag_corpus
        logger.info(f"Added RAG corpus to agent_kwargs: {len(rag_corpus)} documents")

    if args.method == "lmlm":
        # Mirror the two_phase sampling knobs so lmlm runs can be made
        # hyperparameter-identical to a two_phase --use-train-params run.
        agent_kwargs.update({
            "top_p": args.top_p,
            "vllm_top_k": args.vllm_top_k,
            "max_model_len": args.max_model_len,
        })

    agent_kwargs.update(_extra_agent_kwargs)

    # Get agent instance using factory function
    agent: Agent = get_agent(method=args.method, agent_kwargs=agent_kwargs)

    # Build unified database if concat_all_db is enabled (two_phase only)
    unified_db_path = None
    if args.load_unified_db and args.method == "two_phase":
        # Fast path: skip Phase 1 entirely and reconstruct the DatabaseManager
        # from a previously saved unified DB JSON. Only re-runs SentenceTransformer
        # over the stored triplets (a fraction of the original build cost since
        # the LLM triplet-extraction pass is bypassed).
        logger.info(f"Loading prebuilt unified database from {args.load_unified_db}")
        with open(args.load_unified_db, "r", encoding="utf-8") as f:
            db_data = json.load(f)
        loaded_meta = db_data.get("metadata", {})
        loaded_triplets = [
            (t["head"], t["relation"], t["tail"])
            for t in db_data.get("triplets", [])
        ]
        logger.info(
            f"Loaded {len(loaded_triplets)} triplets (built from {loaded_meta.get('num_examples', '?')} examples, dataset={loaded_meta.get('dataset', '?')})"
        )

        unified_db = build_databases_from_triplets_batch(
            [loaded_triplets],
            top_k=agent.top_k,
            default_threshold=agent.similarity_threshold,
            adaptive=False,
            use_inverses=agent.use_inverses,
        )[0]

        # Store in agent the same way the build path does, so Phase 2 finds it.
        agent._unified_db = unified_db
        agent._phase1_info = [{"triplets": loaded_triplets}]
        agent._unified_db_stats = {
            "total_triplets": len(loaded_triplets),
            "num_examples": loaded_meta.get("num_examples", -1),
            "loaded_from": args.load_unified_db,
        }
        unified_db_path = args.load_unified_db
        args.unified_db_path = unified_db_path
        logger.info("Unified database ready (loaded from disk)")
    elif args.concat_all_db and args.method == "two_phase":
        logger.info("Building unified database from entire dataset...")
        logger.info(f"use_contexts={args.use_contexts}, contexts_are_split={args.use_contexts == 'all'}")

        # Prepare all queries and contexts from the full dataset
        all_queries = []
        all_contexts = []

        for ex in full_dataset.select(range(args.start_index, end_index)):
            all_queries.append(ex["question"])
            contexts = ex["contexts"]

            # Apply same context logic as in process_single_batch
            if args.use_contexts == "golden":
                contexts = ex["golden_contexts"]
            elif args.use_contexts == "all":
                # TriviaQA uses custom sentence-based splitting to avoid token limits
                if args.dataset.lower() in {"trivia_qa", "triviaqa", "popqa"}:
                    # Get context_titles from the example (parallel to contexts)
                    context_titles = ex.get("context_titles", [])
                    # Split articles into sentence chunks, then wrap each chunk as its own part
                    chunks = split_trivia_qa_contexts(contexts, context_titles, min_chunk_length=800)
                    contexts = [[chunk] for chunk in chunks]  # Each chunk becomes its own part
                else:
                    # TODO: Check if 5 parts is appropriate for musique, hotpot, and 2wiki dataset sizes
                    contexts = split_into_parts(contexts, num_parts=5)

            all_contexts.append(contexts)

        logger.info(f"Prepared {len(all_queries)} queries for unified DB building")

        # Build the unified database
        if args.dataset == "confiqa":
            # Special handling for ConFiQA: use golden triplets directly, skip phase 1
            logger.info(f"Building ConFiQA database from golden triplets (setting={args.confiqa_setting})")

            all_triplets = []
            for ex in full_dataset.select(range(args.start_index, end_index)):
                triplets = ex.get("golden_triplets", [])
                # Convert triplets to (head, relation, tail) format
                for triplet in triplets:
                    if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                        all_triplets.append(tuple(triplet))

            # Build unified DatabaseManager from golden triplets
            unified_db = build_databases_from_triplets_batch(
                [all_triplets],
                top_k=agent.top_k,
                default_threshold=agent.similarity_threshold,
                adaptive=False,
                use_inverses=agent.use_inverses,
            )[0]

            # Store in agent (same as build_unified_db_from_dataset does)
            agent._unified_db = unified_db
            agent._phase1_info = [{"triplets": all_triplets}]
            agent._unified_db_stats = {
                "total_triplets": len(all_triplets),
                "num_examples": len(all_queries),
            }
            logger.info(f"Built ConFiQA database with {len(all_triplets)} golden triplets")
        else:
            # Normal phase 1 processing for other datasets
            agent.build_unified_db_from_dataset(all_queries, all_contexts)

        logger.info("Unified database built successfully")

        # Save the unified database to disk
        unified_db_dir = os.path.join(output_dir, "unified_databases")
        os.makedirs(unified_db_dir, exist_ok=True)

        contexts_suffix = f"_{args.use_contexts}" if args.use_contexts != "golden" else ""
        confiqa_suffix = (
            f"_{args.confiqa_setting}" if args.dataset == "confiqa" else ""
        )
        unified_db_filename = f"unified_db_{args.dataset}_{args.split}_n{examples_to_process}_i{args.start_index}{contexts_suffix}{confiqa_suffix}.json"
        unified_db_path = os.path.join(unified_db_dir, unified_db_filename)

        # Extract triplets from the unified database and save
        phase1_info = getattr(agent, "_phase1_info", [])
        all_triplets = []
        for info in phase1_info:
            all_triplets.extend(info.get('triplets', []))

        unified_db_data = {
            "metadata": {
                "dataset": args.dataset,
                "split": args.split,
                "num_examples": len(all_queries),
                "start_index": args.start_index,
                "total_triplets": len(all_triplets),
                "use_contexts": args.use_contexts,
                "contexts_are_split": args.use_contexts == "all",
                "dataset_provenance": args._dataset_provenance,
            },
            "triplets": [{"head": t[0], "relation": t[1], "tail": t[2]} for t in all_triplets]
        }

        with open(unified_db_path, "w", encoding="utf-8") as f:
            json.dump(unified_db_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved unified database to {unified_db_path}")
        # Store the path for metadata
        args.unified_db_path = unified_db_path

        if args.phase1_only:
            logger.info(
                "--phase1-only set; unified DB saved (%d triplets from %d examples). "
                "Exiting before Phase 2 as requested.",
                len(all_triplets),
                len(all_queries),
            )
            return

    # Process batches with progress tracking
    # Start with existing results if resuming
    all_results = existing_results.copy() if existing_results else {}
    successful_batches = 0
    failed_batches = 0

    with tqdm(total=batches_to_process, desc="Processing batches", unit="batch") as pbar:
        for batch_num in range(1, batches_to_process + 1):
            try:
                batch_results = process_single_batch(
                    args, batch_num, end_index, full_dataset, agent, all_results
                )
                all_results.update(batch_results)
                if batch_results:  # Only count as successful if we actually processed
                    successful_batches += 1

                # Save based on save_every flag
                if args.save_every and successful_batches % args.save_every == 0:
                    save_results_to_file(all_results, save_path, args)

                pbar.update(1)
            except Exception as e:
                failed_batches += 1
                logger.error("Error processing batch %d: %s", batch_num, e, exc_info=True)
                pbar.update(1)
                raise
                
    logger.info(
        "Completed %d/%d batches successfully (%d failed)",
        successful_batches,
        batches_to_process,
        failed_batches,
    )

    # Save final results (either first time if save_every=-1, or final update)
    save_results_to_file(all_results, save_path, args)

    if args.eval:
        # Evaluate
        results = evaluate_file(
            save_path,
            dataset=args.dataset,
            setting=args.setting,
            split=args.split,
            source='hf',
        )
        logging.info(f"Evaluation results: {json.dumps(results, indent=2)}")


        outpath = save_results(results, "./", save_results_path)
        logging.info(f"Evaluation results saved to: {outpath}")



if __name__ == "__main__":
    main()
