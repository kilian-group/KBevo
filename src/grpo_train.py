from dataclasses import dataclass, field
from typing import Optional
from trainer.lmlm_basetrainer import LMLMGRPOTrainer, parse_triplets
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl.trainer.grpo_config import GRPOConfig
from reward_func import em_accuracy, f1_reward, db_coverage_reward, db_size_threshold, format_reward_zero_rl
import json
import wandb
import os
from data import get_dataset


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    """Dataset and path arguments."""
    model_path: str = field(metadata={"help": "Path to the pretrained model"})
    database_path: str = field(metadata={"help": "Path to the LMLM database JSON file"})

    dataset_name: str = field(
        default="hotpotqa/hotpot_qa",
        metadata={"help": "HuggingFace dataset name"}
    )
    dataset_config: str = field(
        default="distractor",
        metadata={"help": "Dataset configuration"}
    )
    train_size: int = field(default=8000, metadata={"help": "Number of training examples"})
    eval_size: int = field(default=100, metadata={"help": "Number of evaluation examples"})


@dataclass
class LMLMArguments:
    """Core LMLM arguments."""
    retrieval_threshold: float = field(
        default=0.6,
        metadata={
            "help": "Cosine similarity threshold for DB retrieval",
            "aliases": ["--retrieval-threshold"],
        }
    )
    retrieval_top_k: int = field(
        default=1,
        metadata={
            "help": "Maximum number of results retrieved from the DB (separate from generation top_k)",
            "aliases": ["--retrieval-top-k"],
        }
    )
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Wrap prompts in the model's chat template before generation"}
    )

    # --- Core: two-phase training ---
    two_phase: bool = field(
        default=False,
        metadata={"help": "Enable two-phase generation (Phase 1: DB creation, Phase 2: QA with DB)"}
    )
    reward_func: str = field(
        default="em_coverage",
        metadata={"help": "Reward functions to use, e.g. 'em', 'f1', 'coverage', 'size', 'format'"}
    )

    # --- Core: Phase 1 hyperparameters (two_phase only) ---
    phase1_reward_type: str = field(
        default="binary",
        metadata={"help": "Phase 1 reward type: 'binary' or 'utilization'"}
    )
    phase1_prompt_type: str = field(
        default="sft",
        metadata={"help": "Prompt type for Phase 1, e.g. 'sft', 'zero_rl', 'formatted_zero_rl'"}
    )
    phase1_db_weight_mode: str = field(
        default="count_dynamic",
        metadata={"help": "How to weight Phase 1 advantage: 'none' | 'fixed' | 'dynamic' | 'count' | 'count_dynamic'"}
    )
    num_db_rollouts: int = field(
        default=1,
        metadata={"help": "DB rollouts per question (K). num_generations must be divisible by K."}
    )


@dataclass
class AblationArguments:
    """Ablation and experimental arguments (not part of the core method)."""
    # --- Tier filtering ---
    tier_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to per-example rollout-score tier JSON ({qid: int in [1,7]}). Filters training examples by score. This artifact is not shipped with the release."}
    )
    tier_min_score: int = field(default=1, metadata={"help": "Min rollout score (inclusive) to keep."})
    tier_max_score: int = field(default=7, metadata={"help": "Max rollout score (inclusive) to keep."})

    # --- Curriculum learning ---
    curriculum: bool = field(
        default=False,
        metadata={"help": "Enable curriculum learning (requires tier_path and max_steps > 0)."}
    )
    curriculum_phases: str = field(
        default="5-7,3-7,1-7",
        metadata={"help": "Score ranges per curriculum phase, e.g. '5-7,3-7,1-7'."}
    )
    curriculum_steps: str = field(
        default="0.33,0.67",
        metadata={"help": "Fractional step boundaries for phase transitions, e.g. '0.33,0.67'."}
    )

    # --- DB ablations ---
    adaptive_k: bool = field(
        default=False,
        metadata={
            "help": "Use adaptive k for DB retrieval",
            "aliases": ["--adaptive-k"],
        },
    )
    use_inverses: bool = field(
        default=False,
        metadata={
            "help": "Augment DB with inverse triplets (e,r,v) -> (v,r,e)",
            "aliases": ["--use-inverses"],
        }
    )
    vanilla_grpo: bool = field(
        default=False,
        metadata={"help": "Treat (db_g, qa_g) as one trajectory with r_db=r_qa. Requires two_phase=True."}
    )

    # --- Generation ablations ---
    tools: bool = field(default=False, metadata={"help": "Enable tool calling"})
    return_triples: bool = field(default=False, metadata={"help": "Return triples for tool calling"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def process_example(example):
    """Process dataset example into prompt-solution format."""
    return {
        "prompt": f"Question:\n{example['question']}\nAnswer:\n",
        "question": example["question"],
        "contexts": example.get("golden_contexts", []),
        "solution": example["answers"][0]
    }


def _maybe_presave_db_tokens(phase1_prompt_type, model_path, output_dir, tokenizer):
    """Pre-save model with DB special tokens so vLLM sees the correct vocab size at startup.

    vLLM fixes org_vocab_size from the on-disk config at startup; if the training model
    has a larger embedding the _move_model_to_vllm weight-sync will fail with an assertion
    error. Pre-saving ensures both sides agree before the trainer initialises colocated vLLM.
    """
    if phase1_prompt_type != "formatted_zero_rl_v6":
        return model_path
    from multi_lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN
    db_tokens = [DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN]
    to_add = [t for t in db_tokens if t not in tokenizer.get_vocab()]
    if not to_add:
        return model_path
    print(f"formatted_zero_rl_v6: registering {len(to_add)} DB special tokens and pre-saving resized model...")
    tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    presave_path = os.path.join(output_dir, "init_with_db_tokens")
    os.makedirs(presave_path, exist_ok=True)
    tmp_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tmp_model.resize_token_embeddings(len(tokenizer))
    tmp_model.save_pretrained(presave_path)
    tokenizer.save_pretrained(presave_path)
    del tmp_model
    print(f"  Saved to {presave_path} (new vocab size: {len(tokenizer)})")
    return presave_path


def _build_curriculum_schedule(ablation_args, train_dataset, tier_data, max_steps):
    """Build curriculum schedule from tier data. Returns schedule dict or None."""
    if not ablation_args.curriculum:
        return None
    if max_steps <= 0:
        raise ValueError("curriculum requires max_steps > 0 in GRPOConfig")

    score_lookup = {qid: int(item["score"]) for qid, item in tier_data["results"].items()}

    phase_ranges = []
    for p in ablation_args.curriculum_phases.split(","):
        lo, hi = map(int, p.strip().split("-"))
        phase_ranges.append((lo, hi))

    fractions = [float(f) for f in ablation_args.curriculum_steps.split(",")]
    boundaries = [0.0] + fractions + [1.0]
    if len(boundaries) - 1 != len(phase_ranges):
        raise ValueError(
            f"curriculum_steps has {len(fractions)} values but curriculum_phases has "
            f"{len(phase_ranges)} phases; need len(curriculum_steps) == len(curriculum_phases) - 1"
        )

    phase_index_lists = []
    for lo, hi in phase_ranges:
        indices = [i for i, ex in enumerate(train_dataset) if lo <= score_lookup.get(ex["id"], -1) <= hi]
        phase_index_lists.append(indices)
        print(f"  Curriculum phase {lo}-{hi}: {len(indices)} examples")

    phase_step_counts = [
        round((boundaries[i + 1] - boundaries[i]) * max_steps)
        for i in range(len(phase_ranges))
    ]
    print(f"  Curriculum step schedule: {list(zip([f'{lo}-{hi}' for lo, hi in phase_ranges], phase_step_counts))}")
    return {"phase_index_lists": phase_index_lists, "phase_step_counts": phase_step_counts}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ScriptArguments, LMLMArguments, AblationArguments, GRPOConfig))
    script_args, lmlm_args, ablation_args, grpo_config = parser.parse_args_into_dataclasses()

    grpo_config.run_name = os.path.basename(grpo_config.output_dir)
    os.makedirs(grpo_config.output_dir, exist_ok=True)
    if wandb.run is not None:
        wandb.run.name = grpo_config.run_name
    else:
        print("Wandb run is not initialized, skipping wandb name setting")

    # Load dataset
    print(f"Loading dataset: {script_args.dataset_name}")
    train_dataset = get_dataset(name=script_args.dataset_name, setting=script_args.dataset_config, split="train", sub_split="train", limit=script_args.train_size, seed=42)
    test_dataset  = get_dataset(name=script_args.dataset_name, setting=script_args.dataset_config, split="train", sub_split="eval",  limit=script_args.eval_size,  seed=42)

    # Tier filtering (ablation)
    curriculum_schedule = None
    if ablation_args.tier_path is not None:
        print(f"Applying tier filter from {ablation_args.tier_path} (score {ablation_args.tier_min_score}..{ablation_args.tier_max_score})")
        with open(ablation_args.tier_path, "r", encoding="utf-8") as f:
            tier_data = json.load(f)
        valid_ids = {
            qid for qid, item in tier_data["results"].items()
            if ablation_args.tier_min_score <= int(item["score"]) <= ablation_args.tier_max_score
        }
        before = len(train_dataset)
        train_dataset = train_dataset.filter(lambda ex: ex["id"] in valid_ids)
        print(f"Tier filter: {before} -> {len(train_dataset)} examples")
        curriculum_schedule = _build_curriculum_schedule(ablation_args, train_dataset, tier_data, grpo_config.max_steps)

    train_set = train_dataset.map(process_example)
    eval_set  = test_dataset.map(process_example)
    print(f"Train set size: {len(train_set)}, Eval set size: {len(eval_set)}")

    # Load tokenizer (and optionally pre-save with DB special tokens)
    print(f"Loading tokenizer from: {script_args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    script_args.model_path = _maybe_presave_db_tokens(
        lmlm_args.phase1_prompt_type, script_args.model_path, grpo_config.output_dir, tokenizer
    )

    print("GRPO Config:")
    print(f"  Output dir:           {grpo_config.output_dir}")
    print(f"  Run name:             {grpo_config.run_name}")
    print(f"  Num generations:      {grpo_config.num_generations}")
    print(f"  Batch size:           {grpo_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation:{grpo_config.gradient_accumulation_steps}")
    print(f"  Learning rate:        {grpo_config.learning_rate}")
    print(f"  Max grad norm:        {grpo_config.max_grad_norm}")
    print(f"  Use vLLM:             {grpo_config.use_vllm}")
    print(f"  Two phase:            {lmlm_args.two_phase}")
    print(f"  Sampling top-k:       {grpo_config.top_k}")
    print(f"  Retrieval threshold:  {lmlm_args.retrieval_threshold}")
    print(f"  Retrieval top-k:      {lmlm_args.retrieval_top_k}")
    print(f"  Adaptive retrieval:   {ablation_args.adaptive_k}")
    print(f"  Use inverses:         {ablation_args.use_inverses}")

    # Build reward functions
    reward_funcs = []
    if "em"       in lmlm_args.reward_func: reward_funcs.append(em_accuracy)
    if "f1"       in lmlm_args.reward_func: reward_funcs.append(f1_reward)
    if "coverage" in lmlm_args.reward_func: reward_funcs.append(db_coverage_reward)
    if "size"     in lmlm_args.reward_func: reward_funcs.append(db_size_threshold)
    if "format"   in lmlm_args.reward_func: reward_funcs.append(format_reward_zero_rl)
    if not reward_funcs:
        reward_funcs = [em_accuracy]

    print("Initializing LMLMGRPOTrainer...")
    trainer = LMLMGRPOTrainer(
        model=script_args.model_path,
        reward_funcs=reward_funcs,
        lmlm_database_path=script_args.database_path,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=grpo_config,
        # Core LMLM args
        retrieval_threshold=lmlm_args.retrieval_threshold,
        retrieval_top_k=lmlm_args.retrieval_top_k,
        use_chat_template=lmlm_args.use_chat_template,
        two_phase=lmlm_args.two_phase,
        phase1_reward_type=lmlm_args.phase1_reward_type,
        phase1_prompt_type=lmlm_args.phase1_prompt_type,
        phase1_db_weight_mode=lmlm_args.phase1_db_weight_mode,
        num_db_rollouts=lmlm_args.num_db_rollouts,
        # Ablation args
        adaptive_k=ablation_args.adaptive_k,
        tools=ablation_args.tools,
        return_triples=ablation_args.return_triples,
        use_inverses=ablation_args.use_inverses,
        vanilla_grpo=ablation_args.vanilla_grpo,
        curriculum_schedule=curriculum_schedule,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
    print("Training completed!")


if __name__ == "__main__":
    main()
