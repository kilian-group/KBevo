# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import atexit
import copy
import inspect
import json
import os
import textwrap
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import datasets
from multi_lmlm.database.database_manager import DatabaseManager, PerExampleRetriever, build_databases_from_triplets_batch
from multi_lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_END_TOKEN, DB_RETRIEVE_TOKEN
import pandas as pd
import torch
import torch.utils.data
import transformers
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_bitsandbytes_available,
    is_wandb_available,
    PreTrainedTokenizer
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_rich_available

from trl.chat_template_utils import add_response_schema, get_training_chat_template, parse_response
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
    prepare_multimodal_messages_vllm,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available#, is_jmespath_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    # shutdown_event_loop_in_daemon,
    split_pixel_values_by_grid,
    split_tensor_dict,
    # start_event_loop_in_daemon,
    unsplit_pixel_values_by_grid,
)
from multi_lmlm.training.utils.utils_metrics import compute_pretrain_mask, set_tokenizer, set_use_special_dblookup_tokens
set_use_special_dblookup_tokens(True)

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

if is_bitsandbytes_available():
    import bitsandbytes as bnb

def parse_triplets(text: str) -> list[tuple[str, str, str]]:
    """Parse triplets from Phase 1 model output.

    Supports three formats:
      1. Pipe-separated:   ``entity | relationship | value``   — one per line (v5).
      2. Parenthesized:    ``(entity, relationship, value)``   — one per line.
      3. Tab-separated:    ``entity\\trelationship\\tvalue``   — one per line.

    Lines that don't match any format are silently skipped.
    Returns a list of (entity, relationship, value) tuples.
    """
    import re

    triplets: list[tuple[str, str, str]] = []
    # Pattern for parenthesized triplets: (entity, relationship, value)
    paren_pattern = re.compile(r"\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+?)\s*\)")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try pipe-separated format first: entity | relationship | value
        if " | " in line:
            parts = line.split(" | ", 2)
            if len(parts) == 3 and all(p.strip() for p in parts):
                triplets.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
                continue

        # Try parenthesized format: (entity, relationship, value)
        match = paren_pattern.search(line)
        if match:
            triplets.append((match.group(1).strip(), match.group(2).strip(), match.group(3).strip()))
            continue

        # Try tab-separated format: entity\trelationship\tvalue
        parts = line.split("\t")
        if len(parts) == 3:
            triplets.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
            continue

        # Try comma-separated format (no parens): entity, relationship, value
        parts = line.split(", ", 2)
        if len(parts) == 3:
            triplets.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

    return triplets # Rollback as dedulication has weird collapsing behavior
    # # Deduplicate while preserving order
    # seen = set()
    # unique_triplets = []
    # for t in triplets:
    #     if t not in seen:
    #         seen.add(t)
    #         unique_triplets.append(t)
    # return unique_triplets


def extract_db_lookup_last(text : str) -> str | None:
    # BUG: does it need to extract all the db_lookup or the last one?
    #Used in _tool_call_loop 
    if DB_START_TOKEN in text and DB_RETRIEVE_TOKEN in text:
        return DB_START_TOKEN + text.split(DB_START_TOKEN)[1].split(DB_RETRIEVE_TOKEN)[0] + DB_RETRIEVE_TOKEN # BUG
        # return DB_START_TOKEN + text.split(DB_START_TOKEN)[1].split(DB_END_TOKEN)[0] + DB_RETRIEVE_TOKEN # BUG
    else:
        return None

logger = get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = str | PreTrainedModel | Callable[[list, list], list[float]]

# What we call a rollout function is a callable that takes prompts (list) and the trainer instance as parameters and
# returns a dict of generation results. Those results must include "prompt_ids", "completion_ids", and "logprobs"
# fields. Any extra fields (per-completion) are forwarded to the reward functions.
RolloutFunc = Callable[[list[str], "GRPOTrainer"], dict[str, Any]]


class CurriculumRepeatSampler(Sampler):
    """
    Like RepeatSampler, but progressively expands the sampling pool through curriculum phases.

    Each call to ``__iter__`` constitutes one epoch through the *current* phase's pool.
    After each epoch the elapsed-step counter advances, and the next epoch may draw from
    a wider pool (the next curriculum phase).

    Args:
        phase_index_lists: list of index lists into the dataset, one per phase.
            Each phase should be a superset of the previous (easy → hard expansion).
        phase_step_counts: number of optimizer steps to spend in each phase.
            The sampler advances to the next phase after this many steps.
        mini_repeat_count: rollouts per question (= ``num_generations``).
        batch_size: unique questions per optimizer step
            (= ``generation_batch_size // num_generations``).
        repeat_count: reuse factor (= ``num_iterations × steps_per_generation``).
        seed: random seed for reproducibility.
    """

    def __init__(
        self,
        phase_index_lists: list[list[int]],
        phase_step_counts: list[int],
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int = 1,
        seed: int | None = None,
    ):
        assert len(phase_index_lists) == len(phase_step_counts), (
            "phase_index_lists and phase_step_counts must have the same length"
        )
        self.phase_index_lists = phase_index_lists
        self.phase_step_counts = phase_step_counts
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.seed = seed
        self._elapsed_steps = 0
        # Cumulative step boundaries: phase i is active while elapsed < _cum_boundaries[i]
        self._cum_boundaries: list[int] = []
        cumsum = 0
        for s in phase_step_counts:
            cumsum += s
            self._cum_boundaries.append(cumsum)
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def _current_phase(self) -> int:
        for i, boundary in enumerate(self._cum_boundaries):
            if self._elapsed_steps < boundary:
                return i
        return len(self.phase_index_lists) - 1

    def __iter__(self):
        phase = self._current_phase()
        indices = list(self.phase_index_lists[phase])
        logger.info(
            "[CurriculumRepeatSampler] epoch start — phase %d/%d, "
            "elapsed_steps=%d, pool_size=%d",
            phase + 1, len(self.phase_index_lists), self._elapsed_steps, len(indices),
        )

        # Shuffle the phase pool
        perm = torch.randperm(len(indices), generator=self._generator).tolist()
        shuffled = [indices[p] for p in perm]

        # Group into full batches
        chunks = [shuffled[i : i + self.batch_size] for i in range(0, len(shuffled), self.batch_size)]
        chunks = [c for c in chunks if len(c) == self.batch_size]

        steps_yielded = 0
        for chunk in chunks:
            for _ in range(self.repeat_count):
                for idx in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield idx
            steps_yielded += 1

        self._elapsed_steps += steps_yielded

    def __len__(self) -> int:
        phase = self._current_phase()
        n_chunks = len(self.phase_index_lists[phase]) // self.batch_size
        return n_chunks * self.batch_size * self.mini_repeat_count * self.repeat_count


class LMLMGRPOTrainer(BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from trl import GRPOTrainer
    from trl.rewards import accuracy_reward
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                   functions can be either synchronous or asynchronous and can also return `None` when the reward is
                   not applicable to those samples. This is useful for multi-task training where different reward
                   functions apply to different types of samples. When a reward function returns `None` for a sample,
                   that reward function is excluded from the reward calculation for that sample. For more details, see
                   [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions that the model can invoke during generation. Each tool should be a
            standard Python function with properly type-hinted arguments and return values, and a Google-style
            docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It receives the list of prompts allocated to the current
            process and the trainer instance. It must return a dict with `"prompt_ids"`, `"completion_ids"`, and
            `"logprobs"` fields. Any other fields are forwarded to the reward functions. This feature is experimental
            and may change or be removed at any time without prior notice.
    """

    _tag_names = ["trl", "grpo"]
    _name = "GRPO"
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        lmlm_database_path  : str,
        adaptive_k : bool = False,
        return_triples : bool = False,
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        use_inverses: bool = False,
        retrieval_threshold : float = 0.6,
        two_phase: bool = False,
        retrieval_top_k: int = 1,
        phase1_reward_type: str = "binary",
        phase1_prompt_type: str = "sft",
        num_db_rollouts: int = 1,
        phase1_db_weight_mode: str = "fixed_1.0",
        use_chat_template: bool = False,
        vanilla_grpo: bool = False,
        curriculum_schedule: dict | None = None,
    ):
        self.retrieval_top_k = retrieval_top_k
        if self.retrieval_top_k < 1:
            raise ValueError(
                f"retrieval_top_k must be at least 1, got {self.retrieval_top_k}"
            )
        self.use_chat_template = use_chat_template
        #LMLM db initialization
        self.retrieval_threshold = retrieval_threshold
        self.use_inverses = use_inverses

        self.return_triples = return_triples
        self.adaptive_k = adaptive_k
        self.phase1_reward_type = phase1_reward_type  # "binary" or "utilization"
        self.phase1_prompt_type = phase1_prompt_type  # "sft" or "with_question"
        self.phase1_db_weight_mode = phase1_db_weight_mode  # "none" | "fixed[_<w>]" | "dynamic" | "count" | "count_dynamic"

        # Two-phase mode: Phase 1 generates triplets from contexts, Phase 2 does QA with per-example DB
        self.two_phase = two_phase
        self.vanilla_grpo = vanilla_grpo
        self.curriculum_schedule = curriculum_schedule
        if vanilla_grpo:
            assert two_phase, "vanilla_grpo requires two_phase=True"
            # In vanilla GRPO, K=G (one DB per trajectory); num_db_rollouts will be
            # overridden to num_generations at generation time when G is known.
            logger.info(
                "vanilla_grpo=True: (db_g, qa_g) treated as single trajectory; "
                "r_db=r_qa; num_db_rollouts will be auto-set to num_generations (K=G, M=1)"
            )
        self.num_db_rollouts = num_db_rollouts  # K: number of DB rollouts per question
        if two_phase:
            prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "prompts", "database_creation.json"
            )
            with open(prompt_path) as f:
                self._phase1_prompt_template = json.load(f)[phase1_prompt_type]["prompt"]
            logger.info("Two-phase mode enabled: loaded Phase 1 prompt template '%s' from %s", phase1_prompt_type, prompt_path)

            # Phase 2 QA prompt template: zero_rl variants include system instructions; default is bare question
            self._phase2_prompt_template = None
            phase2_prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "prompts", "lmlm_agent.json"
            )
            with open(phase2_prompt_path) as f:
                phase2_prompts = json.load(f)
            if phase1_prompt_type in phase2_prompts and phase1_prompt_type != "sft":
                self._phase2_prompt_template = phase2_prompts[phase1_prompt_type]["prompt"]
                logger.info("Loaded Phase 2 prompt template '%s' from %s", phase1_prompt_type, phase2_prompt_path)
            else:
                logger.info("No Phase 2 prompt template for '%s'; using bare QA prompt", phase1_prompt_type)
        else:
            self.db = DatabaseManager()
            self.db.load_database(
                lmlm_database_path,
                top_k=self.retrieval_top_k,
                default_threshold=self.retrieval_threshold,
                adaptive=self.adaptive_k,
                use_inverses=self.use_inverses,
            )

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), truncation_side="left", padding_side="left"
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # For formatted_zero_rl_v6 starting from a base model, the DB special
        # tokens are not yet in the vocabulary — add them and resize embeddings.
        if phase1_prompt_type == "formatted_zero_rl_v6":
            _db_special_tokens = [DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN]
            _tokens_to_add = [t for t in _db_special_tokens if t not in tokenizer.get_vocab()]
            if _tokens_to_add:
                num_added = tokenizer.add_special_tokens({"additional_special_tokens": _tokens_to_add})
                model.resize_token_embeddings(len(tokenizer))
                logger.info(
                    "formatted_zero_rl_v6: added %d DB special tokens and resized model embeddings to %d: %s",
                    num_added, len(tokenizer), _tokens_to_add,
                )

        # Resolve DB special-token IDs.  SFT models have these as single vocab
        # entries; base models tokenize them into multiple subwords.  Guard
        # against the multi-token case so we don't accidentally use the ID of
        # '<' (first subword) as a stop token.
        db_retrieve_ids = tokenizer.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)
        db_end_ids = tokenizer.encode(DB_END_TOKEN, add_special_tokens=False)

        if len(db_retrieve_ids) == 1:
            self.db_retrieve_token_id = db_retrieve_ids[0]
            self.stop_token_ids = [tokenizer.eos_token_id, db_retrieve_ids[0]]
            self.stop_strings = None  # not needed; token-level stop is active
            logger.info("DB_RETRIEVE_TOKEN %r is a single token (id=%d) — using token-level stop",
                        DB_RETRIEVE_TOKEN, db_retrieve_ids[0])
        else:
            self.db_retrieve_token_id = None
            self.stop_token_ids = [tokenizer.eos_token_id]
            self.stop_strings = [DB_RETRIEVE_TOKEN]  # vLLM string-level stop
            logger.warning(
                "DB_RETRIEVE_TOKEN %r is NOT a single token in this tokenizer "
                "(got %d tokens: %s). Using string-level stop instead of stop_token_ids.",
                DB_RETRIEVE_TOKEN, len(db_retrieve_ids), db_retrieve_ids,
            )

        self.db_end_token_id = db_end_ids[0] if len(db_end_ids) == 1 else None
        set_tokenizer(tokenizer)

        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                model_init_kwargs = args.model_init_kwargs or {}
                # Special case for DeepSpeed: requires device_map=None ("auto" fails)
                if args.distributed_state.distributed_type == "DEEPSPEED":
                    model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        self._has_async_reward_funcs = any(asyncio.iscoroutinefunction(func) for func in self.reward_funcs)
        if self._has_async_reward_funcs:
            self.async_reward_loop_thread, self.async_reward_loop, self.async_reward_loop_ready_event = (
                start_event_loop_in_daemon(name="GRPOTrainer-AsyncRewardLoop")
            )
            # wait until the event loop is running in the daemon thread
            self.async_reward_loop_ready_event.wait()
            atexit.register(shutdown_event_loop_in_daemon, self.async_reward_loop_thread, self.async_reward_loop)

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Rollout function
        if rollout_func is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are importing from 'rollout_func', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func

        # Tools
        # if tools:
        #     if not Version(transformers.__version__) >= Version("5.0.0.dev0"):
        #         raise ImportError(
        #             "Using tools with GRPOTrainer requires transformers version 5.0.0 or higher. Please use "
        #             "transformers with `pip install --pre transformers` to use this feature."
        #         )
        #     if not is_jmespath_available():
        #         raise ImportError(
        #             "Using tools with GRPOTrainer requires the jmespath library for response parsing. Please install "
        #             "it with `pip install jmespath` to use this feature."
        #         )
        self.tools = tools
        # At the time of initial implementation, most tokenizers do not have built-in support for response schemas.
        # While waiting for broader adoption, we provide this utility function to manually set the response schema for
        # known chat templates.
        # We need `getattr`` until the base class sets a default None value for response_schema
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.

        # Training arguments
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        if args.loss_type == "sapo" and (args.sapo_temperature_neg is None or args.sapo_temperature_pos is None):
            raise ValueError(
                "When using `sapo` loss, both `sapo_temperature_neg` and `sapo_temperature_pos` must be set."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = args.model_init_kwargs or {}
            # Special case for DeepSpeed: requires device_map=None ("auto" fails)
            # fix BUG: multi-GPU
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                def cast_inputs_to_fp32(module, inputs):
                    # Preserve other positional args and kwargs untouched
                    if not inputs:
                        return inputs
                    return (inputs[0].to(torch.float32),) + inputs[1:]

                original_dtype_local = target_model.lm_head.weight.dtype
                target_model.lm_head = target_model.lm_head.float()
                target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 (intentional for numerical stability of logits)
                    target_model.model.embed_tokens.register_forward_hook(cast_outputs_to_original_dtype)

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        # current_gradient_accumulation_steps is set by transformers.Trainer only inside the
        # training loop; initialize here so eval (prediction_step → _compute_loss) can use it.
        self.current_gradient_accumulation_steps = args.gradient_accumulation_steps
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        # In two-phase mode, Phase 1 generates K DBs per question and Phase 2 generates N QA rollouts.
        B = args.generation_batch_size
        N = args.num_generations
        K = N if self.vanilla_grpo else self.num_db_rollouts
        if self.two_phase:
            self._logs = {
                "phase1_prompt": deque(maxlen=B * K),
                "phase1_completion": deque(maxlen=B * K),
                "phase1_context": deque(maxlen=B * K),
                "generated_db": deque(maxlen=B * K),
                "rewards": defaultdict(lambda bk=B*K: deque(maxlen=bk)),
                "phase1_advantages": deque(maxlen=B * K),
                "answer": deque(maxlen=B * K),
            }
            for i in range(N):
                self._logs[f"phase2_prompt_{i}"] = deque(maxlen=B)
                self._logs[f"phase2_completion_{i}"] = deque(maxlen=B)
                self._logs[f"phase2_advantages_{i}"] = deque(maxlen=B)
                self._logs["rewards"][f"em_accuracy_{i}"] = deque(maxlen=B)

        else:
            self._logs = {
                "prompt": deque(maxlen=args.generation_batch_size),
                "completion": deque(maxlen=args.generation_batch_size),
                "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
                "advantages": deque(maxlen=args.generation_batch_size),
            }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(
                        base_url=base_url, group_port=args.vllm_group_port, connection_timeout=args.vllm_server_timeout
                    )
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                vllm_quantization = None
                if is_bitsandbytes_available():
                    for _, module in model.named_modules():
                        if isinstance(module, bnb.nn.Linear4bit):
                            vllm_quantization = "bitsandbytes"
                            break
                        elif isinstance(module, bnb.nn.Linear8bitLt):
                            raise ValueError("vLLM does not support in-flight 8-bit quantization.")
                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    # BUG: vllm random issue
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    # max_num_seqs=self.args.per_device_train_batch_size + 8,
                    max_model_len=self.args.vllm_max_model_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                    quantization=vllm_quantization,
                    ## BUG: illegal memory access
                    # enforce_eager=True,
                    # enable_prefix_caching=False, # serving feature; off for RL
                    ##
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": self.stop_token_ids,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't
        # work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            cols = ["prompt"]
            if getattr(self, "two_phase", False):
                cols.append("contexts")
            self._signature_columns = cols

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        if self.curriculum_schedule is not None:
            batch_size = self.args.generation_batch_size // self.num_generations
            logger.info(
                "[CurriculumRepeatSampler] initialising — %d phases, step_counts=%s, batch_size=%d",
                len(self.curriculum_schedule["phase_index_lists"]),
                self.curriculum_schedule["phase_step_counts"],
                batch_size,
            )
            return CurriculumRepeatSampler(
                phase_index_lists=self.curriculum_schedule["phase_index_lists"],
                phase_step_counts=self.curriculum_schedule["phase_step_counts"],
                mini_repeat_count=self.num_generations,
                batch_size=batch_size,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                seed=self.args.seed,
            )
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.args.seed,
        )

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        token_type_ids=None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: list[str] | None = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        # Simply gather (if needed) and update each parameter individually.
        if self.is_fsdp_enabled:
            fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
            fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
            if fsdp_version == 1:
                self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
            elif fsdp_version == 2:
                self._sync_fsdp2_params_to_vllm(self.model)
        else:
            for name, param in self.model.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                with gather_if_zero3([param]):
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator    
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        async_funcs_info = []  # async custom functions for asyncio.gather

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                with profiling_context(self, reward_func_name):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            elif asyncio.iscoroutinefunction(reward_func):  # Separate async reward funcs to run them in parallel later
                async_funcs_info.append((i, reward_func, reward_func_name))
            else:
                # Run synchronous reward function
                with profiling_context(self, reward_func_name):
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Execute async custom functions in parallel using asyncio.gather
        if async_funcs_info:

            async def _invoke_async_reward(index, func, func_name):
                with profiling_context(self, func_name):
                    output = await func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output = [r if r is not None else torch.nan for r in output]
                    return index, output

            async def _run_async_funcs():
                coros = [_invoke_async_reward(i, func, func_name) for (i, func, func_name) in async_funcs_info]
                return await asyncio.gather(*coros)

            async_results = asyncio.run_coroutine_threadsafe(_run_async_funcs(), self.async_reward_loop).result()
            for idx, output_reward_func in async_results:
                rewards_per_func[:, idx] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate_single_turn(self, prompts: list, num_rollouts: int = 1):
        """
        Generate completions with num_rollouts for a single turn of a conversation.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Optionally wrap raw-string prompts in the model's chat template
        if self.use_chat_template and prompts and isinstance(prompts[0], str):
            chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            prompts = [
                self.processing_class.apply_chat_template(
                    [{"role": "user", "content": p}],
                    **chat_template_kwargs,
                )
                for p in prompts
            ]
            # Debug: show first prompt after chat template (once only)
            if not getattr(self, '_debug_chat_template_printed', False):
                self._debug_chat_template_printed = True
                print("\n" + "="*80)
                print("[DEBUG] Prompt AFTER chat template (first example):")
                print("-"*80)
                print(prompts[0][:3000])
                print("-"*80)
                print(f"[DEBUG] Prompt length after chat template (chars): {len(prompts[0])}")
                print("="*80 + "\n")

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up(tags=["weights"])
                # Work around for https://github.com/vllm-project/vllm/issues/29341
                self.llm.collective_rpc("reload_weights")

            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if is_conversational({"prompt": prompts[0]}):
                prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

            # In vLLM, tool call arguments must be JSON strings. See https://github.com/vllm-project/vllm/pull/28820
            for prompt in prompts:  # iterate over each conversation
                if is_conversational({"prompt": prompt}):
                    for message in prompt:  # iterate over each message
                        if "tool_calls" in message:  # check if message has tool calls
                            for call in message["tool_calls"]:
                                args = call["function"]["arguments"]
                                if isinstance(args, dict):  # only convert dict → JSON string
                                    call["function"]["arguments"] = json.dumps(args)

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts = gather_object(prompts)

                if self.accelerator.is_main_process:

                    # Deduplicate: prompts may contain num_rollouts duplicates; take unique set and
                    # ask vLLM to produce num_rollouts completions per unique prompt.
                    ordered_set_of_prompts = all_prompts[::num_rollouts]

                    sampling_params = {
                        "n": num_rollouts,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "guided_decoding_regex": self.guided_decoding_regex,
                        "generation_kwargs": self.args.generation_kwargs,
                    }
                    with profiling_context(self, "vLLM.generate"):
                        if self.rollout_func is not None:
                            rollout_prompts = ordered_set_of_prompts
                            if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                                rollout_prompts = [
                                    apply_chat_template(
                                        {"prompt": p}, self.processing_class, **self.chat_template_kwargs
                                    )["prompt"]
                                    for p in rollout_prompts
                                ]
                            output = self.rollout_func(rollout_prompts, self)
                        else:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                output = self.vllm_client.chat(
                                    messages=ordered_set_of_prompts,
                                    **sampling_params,
                                    chat_template_kwargs=self.chat_template_kwargs,
                                    tools=self.tools,
                                    chat_template=self.chat_template,
                                )
                            else:
                                output = self.vllm_client.generate(prompts=ordered_set_of_prompts, **sampling_params)

                        # Extract required fields and collect any extra fields for reward functions
                        required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                        extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_rollouts times
                # TODO: check if this is correct
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_rollouts)]


                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )

                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]


                # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
                extra_fields = {}
                for key, values in all_extra_fields.items():
                    if isinstance(values, list):
                        extra_fields[key] = values[process_slice]
                    else:
                        extra_fields[key] = values

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":

                if self.rollout_func is not None:
                    rollout_prompts = prompts
                    if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                        rollout_prompts = [
                            apply_chat_template(
                                {"prompt": prompt}, self.processing_class, **self.chat_template_kwargs
                            )["prompt"]
                            for prompt in rollout_prompts
                        ]
                    output = self.rollout_func(rollout_prompts, self)
                    required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                    extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                    prompt_ids = output["prompt_ids"]
                    completion_ids = output["completion_ids"]
                    logprobs = output["logprobs"]
                else:
                    generation_kwargs = {
                        "n": num_rollouts,  # vLLM on each GPU generates num_rollouts per prompt in colocate mode
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
                        "stop_token_ids" : self.stop_token_ids,
                    }
                    if self.stop_strings:
                        generation_kwargs["stop"] = self.stop_strings
                        # Match stop_token_ids behavior: include the stop
                        # string in the output so extract_db_lookup_last()
                        # can find DB_RETRIEVE_TOKEN in the completion text.
                        generation_kwargs["include_stop_str_in_output"] = True
                    if self.args.generation_kwargs is not None:
                        generation_kwargs.update(self.args.generation_kwargs)
                    sampling_params = SamplingParams(**generation_kwargs)

                    if self.vllm_tensor_parallel_size > 1:
                        # Gather prompts from all ranks in the TP group and flatten.
                        # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                        orig_size = len(prompts)
                        gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                        all_prompts = [p for sublist in gathered_prompts for p in sublist]
                    else:
                        all_prompts = prompts

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.wake_up(tags=["kv_cache"])

                    with profiling_context(self, "vLLM.generate"):
                        if is_conversational({"prompt": prompts[0]}):
                            all_outputs = self.llm.chat(
                                all_prompts,
                                sampling_params=sampling_params,
                                use_tqdm=False,
                                chat_template_kwargs=self.chat_template_kwargs,
                                tools=self.tools,
                                chat_template=self.chat_template,
                            )
                        else:
                            all_outputs = self.llm.generate(
                                all_prompts, sampling_params=sampling_params, use_tqdm=False
                            )

                    all_prompt_ids = [output.prompt_token_ids for output in all_outputs for _ in range(num_rollouts)]
                    all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                    all_logprobs = [
                        [next(iter(lp.values())).logprob for lp in output.logprobs]
                        for outputs in all_outputs
                        for output in outputs.outputs
                    ]

                    if self.vllm_tensor_parallel_size > 1:
                        # Slice completions for this rank within its TP group.
                        # Each rank generates all outputs — we keep only our share.
                        local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                        tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                        prompt_ids = all_prompt_ids[tp_slice]
                        completion_ids = all_completion_ids[tp_slice]
                        logprobs = all_logprobs[tp_slice]
                    else:
                        prompt_ids = all_prompt_ids
                        completion_ids = all_completion_ids
                        logprobs = all_logprobs

                    extra_fields = {}  # No extra fields for colocate mode

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.sleep(level=2)

        elif self.use_transformers_paged:
            if is_conversational({"prompt": prompts[0]}):
                processor_outputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    tools=self.tools,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                processor_outputs = self.processing_class(text=prompts)

            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    # Continuous batching API expects 'inputs' arg only
                    all_outputs = unwrapped_model.generate_batch(
                        processor_outputs["input_ids"], generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = processor_outputs["input_ids"]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for paged mode

        else:
            generate_inputs = self.processing_class(
                    text=prompts, padding=True, padding_side="left", return_tensors="pt"
                )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            
            # #trim prompt completion ids until first db lookup
            # print("prompt completion ids :", prompt_completion_ids)

            # prompt_completion_ids = [trim_until_first_lookup(ids) for ids in prompt_completion_ids]


            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for non-rollout_func paths

        return prompt_ids, completion_ids, logprobs, extra_fields

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs, per_example_dbs=None):
        """Multi-turn tool execution loop with database lookups.

        When ``per_example_dbs`` is supplied (two-phase mode, Phase 2), each
        sample uses its own :class:`PerExampleRetriever` built from Phase 1
        triplets.  Otherwise the shared ``self.db`` (global DatabaseManager)
        is used.

        Args:
            per_example_dbs: Optional list of :class:`PerExampleRetriever`
                instances, one per sample. ``None`` → use ``self.db``.
        """
        # Step 1: Initialize tool_mask with 1s for the initial Phase 1 completion tokens (model-generated)
        tool_mask = [[1] * len(cids) for cids in completion_ids]

        tool_calls = [extract_db_lookup_last(completion) for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_call_count = 0
        tool_failure_count = 0

        while idxs_with_tool:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]

            ## Stage 1: DB Injection — call the database and inject results (mask=0)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call = tool_calls[idx]

                prompt_completion_tools[idx] += completions[idx_with_tool]
                tool_call_count += 1
                try:
                    # Use per-example DB (two-phase) or the global shared DB
                    db = per_example_dbs[idx_with_tool] if per_example_dbs is not None else self.db
                    result = ", ".join(db.retrieve_from_database(
                        tool_call, return_triplets=self.return_triples, threshold=self.retrieval_threshold
                    )) + DB_END_TOKEN
                except Exception as e:
                    tool_failure_count += 1
                    result = "unknown" + DB_END_TOKEN

                ## NOTE: p = prompt + query1 + value1, c = query1 + value1, cids = query1, logprobs = query1 + value1
                prompt_completion_tools[idx] += result
                completions[idx_with_tool] += result
                value_ids = list(self.processing_class(result, add_special_tokens=False)["input_ids"])
                completion_ids[idx_with_tool] = list(completion_ids[idx_with_tool]) + value_ids
                tool_mask[idx_with_tool] += [0] * len(value_ids)  # DB-injected tokens: not trainable
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * len(value_ids)

            # Tokenize and filter samples whose length exceeds max allowed length. This is important, because both
            # vLLM and transformers will error out if the input is longer than the model's max length.
            
            ## TODO: for debug. comment out
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.llm.llm_engine.model_config.max_model_len
            elif not self.use_vllm:
                max_model_len = self.model.config.max_position_embeddings
            else:
                raise NotImplementedError(
                    f"Unsupported mode detected: use_vllm={self.use_vllm}, vllm_mode={self.vllm_mode}"
                )

            overlong = [
                len(prompt_ids[idx]) + len(completion_ids[idx]) >= max_model_len
                or len(completion_ids[idx]) >= self.max_completion_length
                for idx in idxs_with_tool
            ]
            assert len(overlong) == len(prompt_completion_tools), (
                f"overlong of length {len(overlong)} and prompt_completion_tools of length "
                f"{len(prompt_completion_tools)} are not the same length"
            )

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if overlong[idx]:
                    prompt_length = len(prompt_ids[idx_with_tool])
                    # NOTE: very brutal truncation.
                    # ct = pct_ids[idx][prompt_length : prompt_length + self.max_completion_length]
                    per_max_completion_length = min(self.max_completion_length, max_model_len - prompt_length)
                    completion_ids[idx_with_tool] = completion_ids[idx_with_tool][:per_max_completion_length]
                    tool_mask[idx_with_tool] = tool_mask[idx_with_tool][:per_max_completion_length]
                    if logprobs is not None:
                        logprobs[idx_with_tool] = logprobs[idx_with_tool][:per_max_completion_length]

            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tools = [pct for pct, o in zip(prompt_completion_tools, overlong, strict=True) if not o]
            if not idxs_with_tool:
                print("all overlong, exit tool loop")
                break

            ## Stage 2: QA Generation — generate post-retrieval reasoning (mask=1)
            prompt_completion_tool_ids, post_tool_ids, post_tool_logprobs, _ = self._generate_single_turn(
                prompt_completion_tools,
                num_rollouts=1
            )

            for i in range(len(post_tool_logprobs)):
                assert len(post_tool_ids[i]) == len(post_tool_logprobs[i]), (
                    f"post_tool_ids of length {len(post_tool_ids[i])} and post_tool_logprobs of length "
                    f"{len(post_tool_logprobs[i])} are not the same length"
                )

            # Sanity check: from experience, this is useful to catch bugs in the chat template
            ## TODO: for debug. comment out
            # for idx in range(len(idxs_with_tool)):
            #     idx_with_tool = idxs_with_tool[idx]
            #     pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
            #     if prompt_ids[idx_with_tool] != pct[: len(prompt_ids[idx_with_tool])]:
            #         raise ValueError(
            #             "The chat template is not prefix-preserving. Please update it to use a prefix-preserving "
            #             "format."
            #         )
            ## TODO: for debug. comment out

            # # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length
            # # NOTE: shit coding logit. move it after update completion_ids with the new completions
            # for idx in range(len(idxs_with_tool)):
            #     idx_with_tool = idxs_with_tool[idx]
            #     prompt_len = len(prompt_ids[idx_with_tool])
            #     # completion_tool_ids = prompt_completion_tool_ids[idx][prompt_len:]
            #     # BUG: protentially mess up the idx and idx_with_tool
            #     excess_length = len(completion_ids[idx_with_tool]) + len(post_tool_ids[idx]) - self.max_completion_length
            #                global                         +      WRONG: post_tool_ids is local-sized
            #     if excess_length > 0:
            #         # If exceeding max length, truncate post_tool_ids
            #         post_tool_ids[idx] = post_tool_ids[idx][:-excess_length]
            #         if logprobs is not None:
            #             post_tool_logprobs[idx] = post_tool_logprobs[idx][:-excess_length]
            #         excess_length = len(completion_ids[idx_with_tool]) + len(post_tool_ids[idx]) - self.max_completion_length
            #         if excess_length > 0:
            #             # If still exceeding max length, truncate completion_tool_ids as well
                    #     prompt_completion_tool_ids[idx_with_tool] = prompt_completion_tool_ids[idx_with_tool][:-excess_length]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            # for idx in range(len(idxs_with_tool)):
            #     idx_with_tool = idxs_with_tool[idx]
                # if logprobs is not None:
                #     logprobs[idx_with_tool] += post_tool_logprobs[idx]

            # Update completion_ids with the new completions (after tool execution)
            ## NOTE: prompt_ids = prompt. prompt_completion_tool_ids = prompt + query1 + value1 + query2. post_tool_ids = query2
            
            # Update completion_ids and tool_mask with Phase 2 tokens
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # prompt_length = len(prompt_ids[idx_with_tool])
                ## NOTE: completion_ids = query1 + value1 + query2
                # completion_ids[idx_with_tool] = prompt_completion_tool_ids[idx][prompt_length:] + post_tool_ids[idx] # BUG: this is wrong due to different tokenization
                completion_ids[idx_with_tool] += post_tool_ids[idx]
                tool_mask[idx_with_tool] += [1] * len(post_tool_ids[idx])  # Model-generated tokens: trainable
                if logprobs is not None:
                    ## NOTE: logprobs = query1 + value1 + query2
                    logprobs[idx_with_tool] += post_tool_logprobs[idx]
                    assert len(logprobs[idx_with_tool]) == len(completion_ids[idx_with_tool]), (
                        f"logprobs of length {len(logprobs[idx_with_tool])} and completion_ids of length "
                        f"{len(completion_ids[idx_with_tool])} are not the same length"
                    )

                # Truncate if exceeding max_completion_length (tool_mask truncated in lockstep)
                if len(completion_ids[idx_with_tool]) > self.max_completion_length:
                    # Compute excess BEFORE truncation so downstream post_tool_ids stays consistent
                    excess = len(completion_ids[idx_with_tool]) - self.max_completion_length
                    completion_ids[idx_with_tool] = completion_ids[idx_with_tool][:self.max_completion_length]
                    tool_mask[idx_with_tool] = tool_mask[idx_with_tool][:self.max_completion_length]
                    if logprobs is not None:
                        logprobs[idx_with_tool] = logprobs[idx_with_tool][:self.max_completion_length]
                    # Also truncate post_tool_ids for consistent downstream decoding
                    if excess >= len(post_tool_ids[idx]):
                        post_tool_ids[idx] = []
                        if post_tool_logprobs is not None:
                            post_tool_logprobs[idx] = []
                    else:
                        post_tool_ids[idx] = post_tool_ids[idx][:-excess]
                        if post_tool_logprobs is not None:
                            post_tool_logprobs[idx] = post_tool_logprobs[idx][:-excess]

            # Only re-decode completions that were modified (idxs_with_tool) to avoid decoding the
            # full B*N batch every loop iteration. post_tool_ids is already local-sized (idxs_with_tool).
            for i, idx_with_tool in enumerate(idxs_with_tool):
                completions[idx_with_tool] = self.processing_class.decode(
                    completion_ids[idx_with_tool], skip_special_tokens=False
                )
            post_tool_completions = [
                self.processing_class.decode(post_tool_ids[i], skip_special_tokens=False) if post_tool_ids[i] else ""
                for i in range(len(post_tool_ids))
            ]

            # # Add post-tool completions to the existing completions
            # for idx in range(len(idxs_with_tool)):
            #     idx_with_tool = idxs_with_tool[idx]
            #     if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
            #         ## NOTE: completions = query1 + value1 + query2
            #         completions[idx_with_tool] += (post_tool_completions[idx])

            # Check for further tool calls in the Phase 2 completions
            tool_calls = [extract_db_lookup_last(completion) for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]

        # Validate lengths
        for i in range(len(completion_ids)):
            assert len(tool_mask[i]) == len(completion_ids[i]), (
                f"tool_mask of length {len(tool_mask[i])} and completion_ids of length "
                f"{len(completion_ids[i])} are not the same length"
            )
        if logprobs is not None:
            for i in range(len(completion_ids)):
                assert len(logprobs[i]) == len(completion_ids[i]), (
                    f"logprobs of length {len(logprobs[i])} and completion_ids of length "
                    f"{len(completion_ids[i])} are not the same length"
                )
        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    def _generate_two_phase(self, qa_prompts: list[str], contexts: list[list[str]], questions: list[str] | None = None, fast_build_db: bool = None):
        """Two-phase generation: Phase 1 (triplet gen) → Phase 2 (QA with per-example DB).

        Phase 1: Model generates knowledge triplets from context paragraphs.
                 Output is parsed into per-example PerExampleRetriever instances.
        Phase 2: Model answers questions using DB lookups against the per-example
                 databases built in Phase 1.

        Args:
            qa_prompts: List of QA prompts
            contexts: List of context lists
            fast_build_db: If True, uses fast database building for eval. Not used in training.
                          Defaults to None.

        Returns:
            Tuple of (prompt_ids, completion_ids, tool_mask, completions,
            logprobs, tool_call_count, tool_failure_count, extra_fields). For both phases.
        """
        # Zero-RL: override bare QA prompts with full system-instruction template
        if self._phase2_prompt_template is not None and questions is not None:
            qa_prompts = [self._phase2_prompt_template.replace("{question}", q) for q in questions]

        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        N = num_generations
        if self.vanilla_grpo:
            # K=G=N: one DB per trajectory, one QA per DB (M=1)
            K = N
            M = 1
            print(f"[vanilla_grpo] K=N={N}, M=1: each of the {N} trajectories gets 1 DB and 1 QA rollout")
        else:
            K = self.num_db_rollouts  # DB rollouts per question
            assert N % K == 0, f"[TRR++] num_generations ({N}) must be divisible by num_db_rollouts ({K})"
            M = N // K  # QA rollouts per (question, DB) pair


        # ===== Phase 1: Generate K triplet DBs per question =====
        if "{question}" in self._phase1_prompt_template and questions is not None:
            phase1_prompts = [
                self._phase1_prompt_template.format(context="\n\n".join(ctx_list), question=q)
                for ctx_list, q in zip(contexts, questions)
            ]
        else:
            phase1_prompts = [
                self._phase1_prompt_template.format(context="\n\n".join(ctx_list))
                for ctx_list in contexts
            ]

        # ── Debug: show first Phase 1 and Phase 2 prompt (before chat template) ──
        if not getattr(self, '_debug_two_phase_prompts_printed', False):
            self._debug_two_phase_prompts_printed = True
            print("\n" + "="*80)
            print("[DEBUG] Phase 1 prompt (first example, before chat template):")
            print("-"*80)
            print(phase1_prompts[0][:3000])
            print("-"*80)
            print(f"[DEBUG] Phase 1 prompt length (chars): {len(phase1_prompts[0])}")
            print(f"[DEBUG] Phase 2 prompt (first example, before chat template):")
            print("-"*80)
            print(qa_prompts[0][:3000])
            print("-"*80)
            print(f"[DEBUG] Phase 2 prompt length (chars): {len(qa_prompts[0])}")
            print(f"[DEBUG] use_chat_template={self.use_chat_template}")
            print("="*80 + "\n")

        (
            phase1_prompt_ids,
            phase1_completion_ids,
            phase1_logprobs,
            extra_fields,
        ) = self._generate_single_turn(phase1_prompts, num_rollouts=K)


        phase1_completions = self.processing_class.batch_decode(
            phase1_completion_ids, skip_special_tokens=False
        )

        phase1_completions_for_db = self.processing_class.batch_decode(
            phase1_completion_ids, skip_special_tokens=True
        )

        # ── Debug: show first Phase 1 completion ──
        if not getattr(self, '_debug_phase1_completion_printed', False) and phase1_completions:
            self._debug_phase1_completion_printed = True
            print("\n" + "="*80)
            print("[DEBUG] Phase 1 COMPLETION (first example, with special tokens):")
            print("-"*80)
            print(phase1_completions[0][:3000])
            print("-"*80)
            print(f"[DEBUG] Phase 1 completion length (tokens): {len(phase1_completion_ids[0])}")
            print("="*80 + "\n")

        B = len(qa_prompts)
        # [TRR++] Phase 1 must produce exactly B*K completions (K per question)
        assert len(phase1_completions) == B * K, (
            f"[TRR++] Phase 1: expected {B * K} completions (K={K} per question), got {len(phase1_completions)}"
        )
        assert len(phase1_completion_ids) == B * K, (
            f"[TRR++] Phase 1: expected {B * K} completion_ids, got {len(phase1_completion_ids)}"
        )
        print(f"[TRR++] Phase 1: B={B}, K={K}, N={N}, M={M} | generated {len(phase1_completions)} DB completions (expected B*K={B*K})")

        logger.info(
            "Phase 1: generated %d completions", len(phase1_completions)
        )

        # Parse triplets from all B*K Phase 1 completions
        all_triplets = []
        total_triplets = 0
        context_lengths = []
        for i, comp_text in enumerate(phase1_completions_for_db):
            triplets = parse_triplets(comp_text)
            total_triplets += len(triplets)
            all_triplets.append(triplets)

            # Calculate context length for this example (K completions share the same context)
            context_length = sum(len(ctx) for ctx in contexts[i // K])
            context_lengths.append(context_length)


        if fast_build_db:
            return [t for triplet_list in all_triplets for t in triplet_list]



        # Log triplet to context ratio
        # print(f"[DEBUG] Triplet to context character ratio for first 5 examples:")
        # for i in range(min(5, len(all_triplets))):
        #     ratio = len(all_triplets[i]) / context_lengths[i] if context_lengths[i] > 0 else 0
        #     print(f"  Example {i}: {len(all_triplets[i])} triplets / {context_lengths[i]} chars = {ratio:.6f} triplets/char")
        #     print(f"    (Context has {len(contexts[i])} paragraphs)")

        # Build per-example DBs in batch (one embedding pass for all triplets)
        per_example_dbs = build_databases_from_triplets_batch(
            all_triplets,
            top_k=self.retrieval_top_k,
            default_threshold=self.retrieval_threshold,
            adaptive=self.adaptive_k,
            use_inverses=self.use_inverses
        )

        self._generated_db_triplet_list = [str(db.database["triplets"]) for db in per_example_dbs]

        logger.info(
            "Phase 1: parsed %d total triplets across %d examples",
            total_triplets,
            len(per_example_dbs),
        )

        DB_SIZE_RATIO_THRESHOLD = 0.005  # matches db_size_threshold reward func threshold

        # [TRR++] B*K per-question DBs (K per question), then expanded to B*N for Phase 2 tool calls
        assert len(per_example_dbs) == B * K, (
            f"[TRR++] expected {B * K} per_example_dbs, got {len(per_example_dbs)}"
        )
        # Expand B*K DBs to B*N: each QA rollout gets a shallow copy with its own _queried_pairs
        # so per-rollout utilization can be tracked independently.
        per_example_dbs_expanded = []
        db_size_floor_expanded = []  # per-rollout denominator floor; prevents reward hacking via tiny DBs
        for b in range(B):
            for i in range(N):
                original_db = per_example_dbs[b * K + i // M]
                db_copy = copy.copy(original_db)
                db_copy._queried_pairs = set()
                per_example_dbs_expanded.append(db_copy)
                db_size_floor_expanded.append(context_lengths[b * K + i // M] * DB_SIZE_RATIO_THRESHOLD)
        assert len(per_example_dbs_expanded) == B * N, (
            f"[TRR++] expected {B * N} expanded DBs, got {len(per_example_dbs_expanded)}"
        )
        print(f"[TRR++] Built {B*K} per-example DBs (B={B}, K={K}), expanded to {len(per_example_dbs_expanded)} for B*N={B*N} QA rollouts")

        # ===== Phase 2: QA with per-example DB lookups =====
        (
            prompt_ids,
            completion_ids,
            logprobs,
            extra_fields,
        ) = self._generate_single_turn(qa_prompts, num_rollouts=N)

        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=False
        )

        # [TRR++] Phase 2 must produce exactly B*N completions
        assert len(completion_ids) == B * N, (
            f"[TRR++] Phase 2: expected {B * N} completions (B={B} * N={N}), got {len(completion_ids)}"
        )
        assert len(prompt_ids) == B * N, (
            f"[TRR++] Phase 2: expected {B * N} prompt_ids, got {len(prompt_ids)}"
        )
        print(f"[TRR++] Phase 2: generated {len(completion_ids)} QA completions (expected B*N={B*N})")

        # Run tool call loop with per-example DBs
        # Expand qa_prompts from B to B*N to match completions
        qa_prompts_expanded = [p for p in qa_prompts for _ in range(num_generations)]
        assert len(qa_prompts_expanded) == B * N, (
            f"[TRR++] expected {B * N} qa_prompts_expanded, got {len(qa_prompts_expanded)}"
        )
        if self.tools:
            (
                tool_mask,
                completions,
                completion_ids,
                logprobs,
                tool_call_count,
                tool_failure_count,
            ) = self._tool_call_loop(
                qa_prompts_expanded,
                prompt_ids,
                completion_ids,
                completions,
                logprobs,
                per_example_dbs=per_example_dbs_expanded,
            )
        else:
            tool_mask = None
            tool_call_count = 0
            tool_failure_count = 0

        # ── Debug: show first Phase 2 completion AFTER tool loop ──
        if not getattr(self, '_debug_phase2_post_tool_printed', False) and completions:
            self._debug_phase2_post_tool_printed = True
            post_tool_text = self.processing_class.decode(completion_ids[0], skip_special_tokens=False) if completion_ids else ""
            print("\n" + "="*80)
            print(f"[DEBUG] Phase 2 FINAL completion (after tool loop, tools={self.tools}, calls={tool_call_count}):")
            print("-"*80)
            print(post_tool_text[:3000])
            print("-"*80)
            print(f"[DEBUG] Phase 2 final completion length (tokens): {len(completion_ids[0])}")
            print("="*80 + "\n")

        # Compute per-rollout triplet utilization ratio (used/total) for logging (and as reward if phase1_reward_type == "utilization")
        # Each QA rollout has its own DB copy, so _queried_pairs reflects only that rollout's queries.
        per_rollout_utilization = []
        for db, db_size_floor in zip(per_example_dbs_expanded, db_size_floor_expanded):
            total = max(len(set(db.database["triplets"])), db_size_floor)  # floor prevents reward hacking via tiny DBs
            used = len(db._queried_pairs)
            ratio = used / total if total > 0 else 0.0
            per_rollout_utilization.append(min(ratio, 1.0))
        self._phase1_utilization = per_rollout_utilization  # (B*N,)
        mean_util = sum(per_rollout_utilization) / max(len(per_rollout_utilization), 1)
        print(f"[TRR++] Phase1 utilization (B*N={B*N}): mean={mean_util:.4f}")

        # # Log Phase 1 stats + per-example retriever diagnostics
        # mode = "train" if self.model.training else "eval"
        # self._metrics[mode]["phase1/total_triplets"].append(total_triplets)
        # self._metrics[mode]["phase1/mean_triplets_per_example"].append(
        #     total_triplets / max(len(per_example_dbs), 1)
        # )
        # # Aggregate fuzzy-match diagnostics across all per-example DBs
        # agg_exact = sum(db._exact_hits for db in per_example_dbs)
        # agg_fuzzy = sum(db._fuzzy_hits for db in per_example_dbs)
        # agg_miss = sum(db._misses for db in per_example_dbs)
        # agg_total = agg_exact + agg_fuzzy + agg_miss
        # self._metrics[mode]["phase2/retrieval_exact_hits"].append(agg_exact)
        # self._metrics[mode]["phase2/retrieval_fuzzy_hits"].append(agg_fuzzy)
        # self._metrics[mode]["phase2/retrieval_misses"].append(agg_miss)
        # if agg_total > 0:
        #     self._metrics[mode]["phase2/retrieval_hit_rate"].append(
        #         (agg_exact + agg_fuzzy) / agg_total
        #     )
        # logger.info(
        #     "Phase 2 retrieval stats: exact=%d, fuzzy=%d, miss=%d (hit rate=%.1f%%)",
        #     agg_exact, agg_fuzzy, agg_miss,
        #     100.0 * (agg_exact + agg_fuzzy) / max(agg_total, 1),
        # )

        # Combine Phase 1 and Phase 2 outputs
        # Phase 1 entries have tool_mask with all 1s (all tokens are model-generated)
        combined_prompt_ids = phase1_prompt_ids + prompt_ids
        combined_completion_ids = phase1_completion_ids + completion_ids
        combined_completions = phase1_completions + completions
        combined_logprobs = phase1_logprobs + logprobs
        # phase1_prompts has B entries; each repeated K times in phase1_completion_ids layout
        combined_prompts = [p for p in phase1_prompts for _ in range(K)] + [p for p in qa_prompts for _ in range(N)]

        # [TRR++] Combined output: B*K Phase-1 + B*N Phase-2 = B*(K+N)
        assert len(combined_completion_ids) == B * K + B * N, (
            f"[TRR++] expected {B*K + B*N} combined completions (B*K={B*K} + B*N={B*N}), got {len(combined_completion_ids)}"
        )
        assert len(combined_prompt_ids) == len(combined_completion_ids), (
            f"[TRR++] combined_prompt_ids/completion_ids length mismatch: {len(combined_prompt_ids)} vs {len(combined_completion_ids)}"
        )
        assert len(combined_prompts) == len(combined_completion_ids), (
            f"[TRR++] combined_prompts length mismatch: {len(combined_prompts)} vs {len(combined_completion_ids)}"
        )
        print(f"[TRR++] Combined: {B*K} Phase-1 + {B*N} Phase-2 = {len(combined_completion_ids)} total (expected {B*K + B*N})")

        phase1_tool_mask = [[1] * len(cids) for cids in phase1_completion_ids]
        # Phase 2 tool masks: from _tool_call_loop if tools enabled, else all 1s. Tool calls should always be enabled for LMLM.
        if tool_mask is not None:
            phase2_tool_mask = tool_mask
        else:
            phase2_tool_mask = [[1] * len(cids) for cids in completion_ids]
        combined_tool_mask = phase1_tool_mask + phase2_tool_mask

        return (
            combined_prompt_ids,
            combined_completion_ids,
            combined_tool_mask,
            combined_completions,
            combined_prompts,
            combined_logprobs,
            tool_call_count,
            tool_failure_count,
            extra_fields,
        )

    def _generate(self, prompts: list, contexts: list[list[str]] | None = None, questions: list[str] | None = None):
        """Generate completions, optionally using two-phase flow.

        When ``self.two_phase`` is True and *contexts* are provided:
          **Phase 1** – generate knowledge triplets from *contexts* using
          ``_phase1_prompt_template``, parse them, and build a lightweight
          :class:`PerExampleRetriever` per sample.
          **Phase 2** – answer the question (original *prompts*) using
          ``_tool_call_loop`` with the per-example databases from Phase 1.

        Only Phase 2 tokens contribute to the GRPO loss; Phase 1 acts as
        a differentiable-through-reward preprocessing step (the reward depends
        on Phase 1 quality because bad triplets ⇒ failed lookups ⇒ wrong answer).

        Args:
            prompts: QA prompts (one per example).
            contexts: Per-example context paragraphs for Phase 1. Ignored
                when ``self.two_phase`` is False.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        # ---------- Two-phase dispatch ----------
        if self.two_phase and contexts is not None:
            (
                prompt_ids,
                completion_ids,
                tool_mask,
                completions,
                combined_prompts,
                logprobs,
                tool_call_count,
                tool_failure_count,
                extra_fields,
            ) = self._generate_two_phase(prompts, contexts, questions=questions)
            prompts = combined_prompts
        else:
            # ---------- Standard single-phase path ----------
            prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts, num_rollouts=1)

            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

            # Extract tool calls from the completions and (possibly) execute them
            if self.tools:
                (
                    tool_mask,
                    completions,
                    completion_ids,
                    logprobs,
                    tool_call_count,
                    tool_failure_count,
                ) = self._tool_call_loop(prompts, prompt_ids, completion_ids, completions, logprobs)
            else:
                tool_mask = None

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:  # count only non-tool tokens (tool_mask=1)
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        if self.tools:
            agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
            tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
            agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
            failure_frequency = (
                (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
            )
            self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        answers = [x["solution"] for x in inputs]

        # Extract contexts for two-phase mode (may be absent for single-phase)
        contexts = None
        questions = None
        if self.two_phase:
            def modify_input(x):
                y = copy.deepcopy(x)
                if "{question}" in self._phase1_prompt_template:
                    y["prompt"] = self._phase1_prompt_template.format(
                        context="\n\n".join(y.get("contexts", [])),
                        question=y.get("question", ""),
                    )
                else:
                    y["prompt"] = self._phase1_prompt_template.format(context="\n\n".join(y.get("contexts", [])))
                return y

            N = self.num_generations if mode == "train" else self.num_generations_eval
            contexts = [x.get("contexts", []) for x in inputs]
            questions = [x.get("question", "") for x in inputs]

            # Deduplicate to B unique questions; _generate_two_phase handles the N rollouts
            # internally (Phase 1: 1 DB_b per question, Phase 2: N a_b_i rollouts per question).
            prompts   = prompts[::N]    # B unique QA prompts
            contexts  = contexts[::N]   # B unique contexts
            questions = questions[::N]  # B unique questions

            K = N if self.vanilla_grpo else self.num_db_rollouts
            # Build inputs list matching B*K + B*N combined completions returned by _generate_two_phase:
            #   Phase 1: B*K inputs (K per question, same contexts — different DB rollouts)
            #   Phase 2: B*N inputs unchanged (one per rollout, for QA scoring)
            phase1_inputs = [modify_input(inputs[i]) for i in range(0, len(inputs), N) for _ in range(K)]
            inputs = phase1_inputs + list(inputs)  # B*K + B*N

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts, contexts=contexts, questions=questions)

        if self.two_phase:
            # combined_prompts from _generate_two_phase = DB_prompts(B) + qa_prompts(B*N)
            prompts = [x["prompt"] for x in inputs]

        assert len(inputs) == len(completions), f"Mismatch: {len(inputs)} inputs vs {len(completions)} completions"
        assert len(prompt_ids_list) == len(completions), f"Mismatch: {len(prompt_ids_list)} prompt_ids vs {len(completions)} completions"

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if self.tools:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")  # 0 for tool result tokens, 1 elsewhere

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if not self.tools else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        **forward_kwargs,  
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            **forward_kwargs,  
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        # NOTE: only affect the logging
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Compute advantages
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        _phase_reorder_idx = None  # set below when two_phase + multi-GPU

        if self.two_phase:
            # Apply weights to each reward function's output and sum.
            # rewards_per_func[:B*K, 0] = NaN  (em_accuracy skips Phase-1 triplets)
            # rewards_per_func[:B*K, 1] = db quality scores for Phase-1
            # rewards_per_func[B*K:, 0] = em_accuracy scores for Phase-2 QA
            # rewards_per_func[B*K:, 1] = NaN  (db quality skips Phase-2)
            rewards = rewards_per_func

            # Multi-GPU fix: gather() concatenates per-process tensors along dim=0, producing
            # [GPU0_phase1 + GPU0_phase2, GPU1_phase1 + GPU1_phase2, ...].
            # Reorder to [all_phase1, all_phase2] so [:B*K] and [B*K:] correctly separate phases.
            num_processes = self.accelerator.num_processes
            if num_processes > 1:
                _K = num_generations if self.vanilla_grpo else self.num_db_rollouts
                chunk = len(rewards) // num_processes                  # B_local * (K + N) per GPU
                local_p1 = chunk * _K // (_K + num_generations)        # B_local * K
                idx = []
                for p in range(num_processes):
                    base = p * chunk
                    idx.extend(range(base, base + local_p1))           # phase-1 rows from GPU p
                for p in range(num_processes):
                    base = p * chunk
                    idx.extend(range(base + local_p1, base + chunk))   # phase-2 rows from GPU p
                _phase_reorder_idx = torch.tensor(idx, device=rewards.device)
                rewards = rewards[_phase_reorder_idx]
                rewards_per_func = rewards_per_func[_phase_reorder_idx]

            # Advantage computation for two-phase modes.
            # rewards shape: (B*K + B*N, num_funcs) — first B*K are Phase-1, next B*N are Phase-2
            N = num_generations
            # vanilla_grpo uses K=N (set during generation); re-derive here for the advantage step
            K = N if self.vanilla_grpo else self.num_db_rollouts
            M = N // K  # QA rollouts per (question, DB) pair; M=1 for vanilla_grpo
            assert len(rewards) % (K + N) == 0, (
                f"[TRR++] rewards length {len(rewards)} not divisible by (K+N)={K + N}; "
                f"expected B*(K+N) for some integer B"
            )
            B = len(rewards) // (K + N)  # B*(K+N) = B*K + B*N
            print(f"[TRR++] Advantage: B={B}, K={K}, N={N}, M={M} | rewards shape={tuple(rewards.shape)} (expected B*K+B*N={B*K + B*N})")

            if self.vanilla_grpo:
                # ── Vanilla GRPO advantage ──────────────────────────────────────────────
                # Each trajectory τ_g = (db_g, qa_g) is treated as a single unit.
                # r_g = r_qa_g; both DB and QA tokens share advantage A_g.
                # A_g = r_g - mean_G(r_g), normalized within each question's G=N trajectories.
                assert K == N and M == 1, f"[vanilla_grpo] expected K=N={N}, M=1; got K={K}, M={M}"
                G = N  # number of full trajectories per question
                r_g = rewards[B*G:, 0].nan_to_num(0.0)   # (B*G,) — QA EM rewards

                r_mat = r_g.view(B, G)                              # (B, G)
                baseline = r_mat.mean(dim=1, keepdim=True)          # (B, 1)
                std_g = r_mat.std(dim=1, keepdim=True)              # (B, 1)
                A_g = (r_mat - baseline).view(B * G)                # (B*G,)

                # DB and QA tokens both receive A_g — same advantage for the whole trajectory
                advantages = torch.cat([A_g, A_g])                 # (B*G + B*G,) = (B*K + B*N,)
                mean_grouped_rewards = torch.cat([r_g, r_g])        # for logging
                std_rewards = std_g.expand(B, G).reshape(B * G).repeat(2)  # (2*B*G,)

                print(f"[vanilla_grpo] r_g: mean={r_g.mean():.4f}, std={r_g.std():.4f} | "
                      f"A_g: mean={A_g.mean():.4f}, std={A_g.std():.4f}")

                assert advantages.shape == (B * K + B * N,), \
                    f"[vanilla_grpo] advantages shape {advantages.shape} != ({B*K + B*N},)"
                assert std_rewards.shape == (B * K + B * N,), \
                    f"[vanilla_grpo] std_rewards shape {std_rewards.shape} != ({B*K + B*N},)"

                is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
                if self.scale_rewards != "none":
                    advantages = advantages / (std_rewards + 1e-4)

                print(f"[vanilla_grpo] Post-norm — A_db: mean={advantages[:B*G].mean():.4f}, std={advantages[:B*G].std():.4f} | "
                      f"A_qa: mean={advantages[B*G:].mean():.4f}, std={advantages[B*G:].std():.4f}")

            else:
                # ── Standard TRR++ two-phase advantage ─────────────────────────────────
                weights = self.reward_weights.to(device)          # [w_em, w_db]
                r_b_k_m   = rewards[B*K:, 0].nan_to_num(0.0)       # (B*K*M,) QA EM scores
                # Always gather utilization for logging (computed unconditionally in _generate_two_phase)
                if hasattr(self, '_phase1_utilization'):
                    all_utilization = gather_object(self._phase1_utilization)
                    self._phase1_utilization_gathered = all_utilization  # cache for logging
                    mean_util = sum(all_utilization) / max(len(all_utilization), 1)
                    self._metrics[mode]["phase1/utilization_mean"].append(mean_util)
                else:
                    self._phase1_utilization_gathered = None

                # --- phase1 DB advantage ---
                # r_db_b_k: composite reward for each (question b, DB k) pair — shape (B*K,)
                if self.phase1_reward_type == "utilization":
                    # rollout_util_b_k_m: per-rollout triplet utilization ratio (used/total), shape (B*K*M,).
                    # Always set by _generate_two_phase before this point.
                    # r_b_k_m: (B*K*M,) per-rollout QA correctness. Element-wise product then mean over M rollouts per (b,k).
                    rollout_util_b_k_m = torch.tensor(self._phase1_utilization_gathered, dtype=torch.float32, device=device)  # (B*K*M,)
                    # r_b_k_m_format = r_b_k_m + 0.1  # even in incorrect rollout, retrieved db has small reward
                    r_b_k_m_format = r_b_k_m # DEBUG
                    r_db_b_k = (r_b_k_m_format * rollout_util_b_k_m).view(B, K, M).mean(dim=2).view(B * K)  # (B*K,)
                else:
                    r_b_k_mean = r_b_k_m.view(B, K, M).mean(dim=2).view(B * K)  # (B*K,)
                    if rewards.shape[1] > 1:
                        db_cov_b_k = rewards[:B*K, 1].nan_to_num(0.0)   # (B*K,) db quality scores from reward func
                        r_db_b_k = r_b_k_mean * weights[0] + db_cov_b_k * weights[1]  # (B*K,) additive
                    else:
                        # Only 1 reward function — no separate db coverage score; use QA mean directly
                        r_db_b_k = r_b_k_mean

                if K == 1:
                    # K=1: batch-level normalization across B questions (original behavior)
                    db_baseline_b    = r_db_b_k.mean()               # scalar
                    db_std_b         = r_db_b_k.std()                # scalar
                    A_db_b_k           = r_db_b_k - db_baseline_b        # (B*K,) = (B,) when K=1
                    std_db_b_expanded = db_std_b.expand(B)             # (B*K,) = (B,) when K=1
                else:
                    # K>1: within-question normalization across K DBs
                    r_db_mat_b_k        = r_db_b_k.view(B, K)
                    db_baseline_b       = r_db_mat_b_k.mean(dim=1, keepdim=True)                          # (B, 1)
                    db_std_b            = r_db_mat_b_k.std(dim=1)                                         # (B,)
                    A_db_b_k            = (r_db_mat_b_k - db_baseline_b).view(B * K)                      # (B*K,)
                    std_db_b_expanded = db_std_b.repeat_interleave(K)                                   # (B*K,)

                # --- phase2 QA advantage: within (b,k) group across M rollouts ---
                # For K=1 (M=N): reduces to standard GRPO per-question normalization
                r_qa_mat_b_k_m       = r_b_k_m.view(B, K, M)
                qa_baseline_b_k = r_qa_mat_b_k_m.mean(dim=2)              # (B, K)
                qa_std_b_k  = r_qa_mat_b_k_m.std(dim=2)               # (B, K)
                A_qa_b_k_m     = (r_qa_mat_b_k_m - qa_baseline_b_k.unsqueeze(2)).view(B * N)             # (B*K*M,)
                std_qa_b_k_expanded = qa_std_b_k.repeat_interleave(M, dim=1).view(B * N)           # (B*N,)

                # --- phase1 vs phase2 weighting ---
                # mode format: "none" | "fixed_<w>" (e.g. "fixed_2.0") | "dynamic" | "count" | "count_dynamic"
                raw_mode = self.phase1_db_weight_mode
                if raw_mode == "none":
                    db_weight = 1.0
                elif raw_mode == "dynamic":
                    db_weight = (r_b_k_m.mean() / (r_db_b_k.mean() + 1e-8)).clamp(max=10).item()
                elif raw_mode == "count":
                    db_weight = float(M)
                elif raw_mode == "count_dynamic":
                    db_weight = M * (r_b_k_m.mean() / (r_db_b_k.mean() + 1e-8)).clamp(max=10).item()
                elif raw_mode.startswith("fixed"):
                    # "fixed" → 1.0, "fixed_2.0" → 2.0
                    db_weight = float(raw_mode[6:]) if len(raw_mode) > 5 else 1.0
                else:
                    raise ValueError(f"Unknown phase1_db_weight_mode: {raw_mode!r}. Choose from: none, fixed, fixed_<w>, dynamic, count, count_dynamic")
                print(f"[TRR++] phase1_db_weight_mode={raw_mode}, db_weight={db_weight:.4f}")
                advantages         = torch.cat([A_db_b_k * db_weight, A_qa_b_k_m])  # (B*K + B*K*M,)
                mean_grouped_rewards = torch.cat([r_db_b_k, r_b_k_m])               # for logging
                std_rewards        = torch.cat([std_db_b_expanded, std_qa_b_k_expanded]) # (B*K + B*N,)

                # Verify shapes
                assert r_b_k_m.shape == (B * N,),      f"[TRR++] r_b_k_m shape {r_b_k_m.shape} != ({B * N},)"
                assert r_db_b_k.shape == (B * K,),   f"[TRR++] r_db_b_k shape {r_db_b_k.shape} != ({B * K},)"
                assert A_db_b_k.shape == (B * K,),        f"[TRR++] A_db_b_k shape {A_db_b_k.shape} != ({B * K},)"
                assert A_qa_b_k_m.shape == (B * N,),  f"[TRR++] A_qa_b_k_m shape {A_qa_b_k_m.shape} != ({B * N},)"
                assert advantages.shape == (B * K + B * N,), f"[TRR++] advantages shape {advantages.shape} != ({B*K + B*N},)"
                assert std_rewards.shape == (B * K + B * N,), f"[TRR++] std_rewards shape {std_rewards.shape} != ({B*K + B*N},)"

                print(f"[TRR++] Number of rewards non-zero for QA: {torch.sum(r_b_k_m != 0).item()}")
                print(f"[TRR++] Number of rewards non-zero for DB: {torch.sum(r_db_b_k != 0).item()}")
                print(
                    f"[TRR++] r_b_k_m (QA): mean={r_b_k_m.mean():.4f} | "
                    f"r_db_b_k: mean={r_db_b_k.mean():.4f}, std={r_db_b_k.std():.4f} | "
                    f"qa_std_b_k: mean={qa_std_b_k.mean():.4f}"
                )
                print(
                    f"[TRR++] Pre-norm — A_db_b_k: mean={A_db_b_k.mean():.4f}, std={A_db_b_k.std():.4f} | "
                    f"A_qa_b_k_m: mean={A_qa_b_k_m.mean():.4f}, std={A_qa_b_k_m.std():.4f}"
                )

                is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
                if self.scale_rewards != "none":
                    advantages = advantages / (std_rewards + 1e-4)

                print(
                    f"[TRR++] Post-norm — A_db_b_k: mean={advantages[:B*K].mean():.4f}, std={advantages[:B*K].std():.4f} | "
                    f"A_qa_b_k_m: mean={advantages[B*K:].mean():.4f}, std={advantages[B*K:].std():.4f}"
                )
        else:
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

            # Standard GRPO advantage computation
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards

            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll still log group level std
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # this case doesn't occur during training, but could in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:  # this case doesn't occur during training, but could in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep in [all_p1, all_p2] order for logging
        # Un-reorder advantages back to per-process ordering so process_slice extracts the correct local data
        if _phase_reorder_idx is not None:
            _inv_idx = torch.empty_like(_phase_reorder_idx)
            _inv_idx[_phase_reorder_idx] = torch.arange(len(_phase_reorder_idx), device=advantages.device)
            advantages = advantages[_inv_idx]
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        if self.two_phase:
            K = num_generations if self.vanilla_grpo else self.num_db_rollouts
            BK = B * K  # number of Phase 1 entries
            all_prompts = gather_object(prompts_text)
            all_completions = gather_object(completions_text)
            # Reorder gathered prompts/completions to [all_p1, all_p2] to match reordered advantages
            if _phase_reorder_idx is not None:
                _reorder = _phase_reorder_idx.tolist()
                all_prompts = [all_prompts[i] for i in _reorder]
                all_completions = [all_completions[i] for i in _reorder]
            all_advantages = all_process_advantages.tolist()
            # Gather answers and contexts across all GPUs so their counts match B_global*K
            _N_local = len(answers) // len(contexts)  # num_generations per question
            all_answers_unique = gather_object([answers[i * _N_local] for i in range(len(contexts))])  # B_global unique
            all_contexts_unique = gather_object(contexts)  # B_global unique contexts
            em_accuracy_rewards_list = rewards_per_func[BK:, 0].tolist()  # Phase 2 EM scores
            if self.phase1_reward_type == "utilization" and getattr(self, '_phase1_utilization_gathered', None) is not None:
                ## TODO: this part is misleading as utilization is different from db_threshold
                # _phase1_utilization_gathered is (B*N,); collapse to (B*K,) by averaging over M rollouts per (b,k)
                N = self.num_generations
                M = N // K
                util_bn = list(self._phase1_utilization_gathered)  # (B*N,)
                db_threshold_rewards_list = (
                    torch.tensor(util_bn, dtype=torch.float32)
                    .view(B, K, M).mean(dim=2).view(B * K).tolist()
                )  # (B*K,)
            else:
                db_threshold_rewards_list = (
                    rewards_per_func[:BK, 1].tolist() if rewards_per_func.shape[1] > 1
                    else [0.0] * BK
                )  # (B*K,)

            phase1_prompts = all_prompts[:BK]
            phase2_prompts = all_prompts[BK:]
            phase1_completions = all_completions[:BK]
            phase2_completions = all_completions[BK:]
            phase1_advantages = all_advantages[:BK]
            phase2_advantages = all_advantages[BK:]

            self._logs["phase1_prompt"].extend(phase1_prompts)
            self._logs["phase1_completion"].extend(phase1_completions)
            self._logs["phase1_advantages"].extend(phase1_advantages)
            self._logs["rewards"]["db_size_threshold"].extend(db_threshold_rewards_list)
            # Repeat context and answer K times so all phase1 log columns have B*K entries
            # Use gathered versions (all_contexts_unique, all_answers_unique) so counts match B_global*K
            self._logs["phase1_context"].extend(["\n\n".join(ctx_list) for ctx_list in all_contexts_unique for _ in range(K)])
            all_generated_dbs = gather_object(self._generated_db_triplet_list)
            self._logs["generated_db"].extend(all_generated_dbs)

            self._logs["answer"].extend([ans for ans in all_answers_unique for _ in range(K)])

            for i in range(N):
                phase2_prompts_i = [phase2_prompts[b*N + i] for b in range(B)]
                phase2_completions_i = [phase2_completions[b*N + i] for b in range(B)]
                phase2_advantages_i = [phase2_advantages[b*N + i] for b in range(B)]
                em_accuracy_rewards_i = [em_accuracy_rewards_list[b*N + i] for b in range(B)]

                self._logs[f"phase2_prompt_{i}"].extend(phase2_prompts_i)
                self._logs[f"phase2_completion_{i}"].extend(phase2_completions_i)
                self._logs[f"phase2_advantages_{i}"].extend(phase2_advantages_i)
                self._logs["rewards"][f"em_accuracy_{i}"].extend(em_accuracy_rewards_i)


        else:
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if not self.tools else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if self.tools:
            output["tool_mask"] = tool_mask

        logger.info("="*80)
        logger.info("OUTPUT DICT SHAPES (before shuffle):")
        logger.info(f"  two_phase mode: {self.two_phase}")
        logger.info(f"  len(prompts): {len(prompts)}")
        logger.info(f"  len(completions): {len(completions)}")
        logger.info(f"  len(inputs): {len(inputs)}")
        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"  {key}: {val.shape}")
            else:
                logger.info(f"  {key}: {type(val)} = {val}")
        logger.info("="*80)

        return output

    def check_length(self, completions, logprobs, idx):
        if isinstance(completions[0], str):
            encoded_prompt_completion_tools = self.processing_class(completions[idx], add_special_tokens = False)["input_ids"]
            if len(encoded_prompt_completion_tools) != len(logprobs[idx]):
                print(f"encoded_prompt_completion_tools of length {len(encoded_prompt_completion_tools)} and logprobs of length {len(logprobs[idx])} are not the same length")
        elif isinstance(completions[0], list) and isinstance(completions[0][0], int):
            encoded_prompt_completion_tools = completions[idx]
            if len(encoded_prompt_completion_tools) != len(logprobs[idx]):
                print(f"completion_ids of length {len(encoded_prompt_completion_tools)} and logprobs of length {len(logprobs[idx])} are not the same length")
            if abs(len(encoded_prompt_completion_tools) - len(logprobs[idx])) >= 5:
                import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
        else:
            pass
            # print("encoded_prompt_completion_tools and logprobs are the same length")


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    @staticmethod
    def get_sapo_token_loss(unclipped_token_loss: torch.Tensor, temperature: float) -> torch.Tensor:
        sigmoid_input = temperature * (unclipped_token_loss - 1)
        sigmoid_smoothed_loss = torch.nn.functional.sigmoid(sigmoid_input)
        sapo_token_loss = sigmoid_smoothed_loss * 4 / temperature
        return sapo_token_loss

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]

        # Detect which completions contain the DB_RETRIEVE special token (for separate loss logging).
        # For base models where DB_RETRIEVE_TOKEN is multi-token, db_retrieve_token_id is None —
        # fall back to assuming all completions are triplets (Phase 1 only).
        if self.db_retrieve_token_id is not None:
            has_special_token = (completion_ids == self.db_retrieve_token_id).any(dim=1)  # (B,)
        else:
            has_special_token = torch.zeros(completion_ids.shape[0], dtype=torch.bool, device=completion_ids.device)
        has_triplets = ~has_special_token


        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps

            # Compute and log separate losses for triplets vs special token completions.
            # IMPORTANT: gather() is a collective — all ranks must call it the same number of times.
            # Use NaN when a rank has no samples for a category so the gather always executes.
            if has_triplets.any():
                triplet_per_token_loss = per_token_loss[has_triplets]
                triplet_mask = mask[has_triplets]
                triplet_loss = ((triplet_per_token_loss * triplet_mask).sum(-1) / triplet_mask.sum(-1).clamp(min=1.0)).mean()
            else:
                triplet_loss = torch.tensor(float("nan"), device=per_token_loss.device)
            self._metrics[mode]["loss/triplets"].append(self.accelerator.gather(triplet_loss).nanmean().item())

            if has_special_token.any():
                special_per_token_loss = per_token_loss[has_special_token]
                special_mask = mask[has_special_token]
                special_loss = ((special_per_token_loss * special_mask).sum(-1) / special_mask.sum(-1).clamp(min=1.0)).mean()
            else:
                special_loss = torch.tensor(float("nan"), device=per_token_loss.device)
            self._metrics[mode]["loss/special_tokens"].append(self.accelerator.gather(special_loss).nanmean().item())
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # Use log keys to determine mode: HF Trainer restores model to train mode before calling log(),
        # so self.model.training is unreliable. Eval calls always have "eval_" prefixed keys.
        mode = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                if not self.two_phase: #not supported for 2 phase.
                    print_prompt_completions_sample(
                        self._logs["prompt"],
                        self._logs["completion"],
                        self._logs["rewards"],
                        self._logs["advantages"],
                        self.state.global_step,
                        self.num_completions_to_print,
                    )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)

            if self.two_phase:
                K = self.num_generations if self.vanilla_grpo else self.num_db_rollouts
                N = self.num_generations
                M = N // K

                phase1_prompts = list(self._logs["phase1_prompt"])
                phase1_completions = list(self._logs["phase1_completion"])
                phase1_contexts = list(self._logs["phase1_context"])
                generated_dbs = list(self._logs["generated_db"])
                phase1_advantages = list(self._logs["phase1_advantages"])
                phase1_answers = list(self._logs["answer"])
                db_thresholds = list(self._logs["rewards"]["db_size_threshold"])

                B = len(phase1_prompts) // K

                # inside log(), two-phase branch, right before building combined_table / df_combined

                num_phase1 = len(self._logs["phase1_prompt"])

                # sanity checks
                assert len(self._logs["phase1_completion"]) == num_phase1
                assert len(self._logs["phase1_context"]) == num_phase1
                assert len(self._logs["generated_db"]) == num_phase1
                assert len(self._logs["phase1_advantages"]) == num_phase1
                assert len(self._logs["answer"]) == num_phase1

                combined_table = {
                    "step": [str(self.state.global_step)] * num_phase1,
                    "phase1_prompt": list(self._logs["phase1_prompt"]),
                    "phase1_completion": list(self._logs["phase1_completion"]),
                    "phase1_context": list(self._logs["phase1_context"]),
                    "generated_db": list(self._logs["generated_db"]),
                    "phase1_advantage": list(self._logs["phase1_advantages"]),
                    "answer": list(self._logs["answer"]),
                }

                for reward_name, reward_vals in self._logs["rewards"].items():
                    reward_vals = list(reward_vals)
                    if len(reward_vals) == num_phase1:
                        combined_table[reward_name] = reward_vals

                df_combined = pd.DataFrame(combined_table)

                # Build phase2 table separately (phase2 has B*N rows vs phase1's B*K rows)
                num_phase2 = len(self._logs["phase2_prompt_0"]) if "phase2_prompt_0" in self._logs else 0
                phase2_table = None
                if num_phase2 > 0:
                    phase2_table = {"step": [str(self.state.global_step)] * num_phase2}
                    for i in range(self.num_generations):
                        key = f"phase2_prompt_{i}"
                        if key in self._logs and len(self._logs[key]) == num_phase2:
                            phase2_table[f"phase2_prompt_{i}"] = list(self._logs[f"phase2_prompt_{i}"])
                            phase2_table[f"phase2_completion_{i}"] = list(self._logs[f"phase2_completion_{i}"])
                            phase2_table[f"phase2_advantage_{i}"] = list(self._logs[f"phase2_advantages_{i}"])
                            if f"em_accuracy_{i}" in self._logs["rewards"]:
                                phase2_table[f"em_accuracy_{i}"] = list(self._logs["rewards"][f"em_accuracy_{i}"])

                for logging_backend in logging_backends:
                    logging_backend.log({
                        "completions": logging_backend.Table(dataframe=df_combined),
                    })
                    if phase2_table is not None:
                        logging_backend.log({
                            "completions_phase2": logging_backend.Table(dataframe=pd.DataFrame(phase2_table)),
                        })
            else:
                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }
                df_base = pd.DataFrame(table)
                for logging_backend in logging_backends:
                    df = df_base
                    if self.log_unique_prompts:
                        df = df.drop_duplicates(subset=["prompt"])
                    logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
