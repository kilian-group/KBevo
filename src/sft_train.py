# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from accelerate import Accelerator
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    get_peft_config,
)
from transformers import DataCollatorWithPadding, default_data_collator
from functools import partial

from multi_lmlm.training.utils.utils_metrics import (
    compute_loss_func,
    set_wandb,
    set_tokenizer,
    compute_metrics,
    set_use_special_dblookup_tokens,
)
from multi_lmlm.training.utils.load_model import initialize_model_for_pretraining, load_model_for_ft_baseline
from multi_lmlm.training.utils.load_sft_dataset import prepare_pretrain_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PretrainConfig:
    use_special_dblookup_tokens: bool = False
    use_all_relationships_token: bool = False
    plain_baseline: bool = False
    eval_only: bool = False
    max_seq_length: int = 2048


def set_random_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, PretrainConfig)
    if subparsers is not None:
        return subparsers.add_parser("pretrain", help="Run LMLM pretraining", dataclass_types=dataclass_types)
    return TrlParser(dataclass_types)

def check(data, collator, tokenizer):
    # After data and collator are loaded
    sample = data[:1]
    batch = collator(sample)
    # Collator usually expects a *list* of samples
    for i in range(1):
        input_ids = batch['input_ids'][i]
        labels = batch['labels'][i]
        attention_mask = batch['attention_mask'][i]

        logger.info(f"shape: {input_ids.shape} {labels.shape} {attention_mask.shape}")

        logger.info("=== Input ===")
        logger.info(tokenizer.decode(input_ids, skip_special_tokens=False))

        logger.info("\n=== Labels ===")
        decoded_labels = []
        for input_id, label in zip(input_ids, labels):
            if label != -100:
                decoded_labels.append(tokenizer.decode([input_id]))
            else:
                decoded_labels.append("[MASKED]")

        logger.info(" ".join(decoded_labels))


def main(script_args, training_args, model_args, pretrain_args):
    accelerator = Accelerator()
    set_random_seed(getattr(training_args, "seed", 42))

    if accelerator.is_main_process:
        set_wandb()

    if pretrain_args.plain_baseline and pretrain_args.use_special_dblookup_tokens:
        raise ValueError("Cannot enable both `plain_baseline` and `use_special_dblookup_tokens`.")

    if accelerator.is_main_process:
        logger.info(f"use_special_dblookup_tokens = {pretrain_args.use_special_dblookup_tokens}")
        logger.info(f"use_all_relationships_token = {pretrain_args.use_all_relationships_token}")

    model, tokenizer = load_model_for_ft_baseline(
        model_args,
        resume_from_checkpoint=training_args.resume_from_checkpoint, # HACK: this is a hack to load the model from the checkpoint
        use_special_dblookup_tokens=pretrain_args.use_special_dblookup_tokens,
        use_all_relationships_token=pretrain_args.use_all_relationships_token,
    )

    # tokenizer.pad_token = tokenizer.eos_token
    
    ################
    # Dataset
    ################
    train_dataset, eval_dataset = prepare_pretrain_data(script_args, pretrain_args.use_special_dblookup_tokens, pretrain_args.plain_baseline)

    def tokenize_fn_raw(example):
        tokenized= tokenizer(
            example["annotated_text"],          # or whatever your input field is
            truncation=True,
            padding="max_length",               # or "longest" for dynamic padding
            max_length=pretrain_args.max_seq_length,                     # or whatever your fine-tune max_len is
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        tokenized["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in tokenized["labels"]] 
        return tokenized

    def tokenize_fn(example, max_length = 2048):
        # old sft data stored the prompt and answer in the same field, under the format "Question:\n{prompt}\nAnswer:\n". This is hard to parse / does not generalize, 
        # so from now can have a seperate field for the prompt and answer. Ideally the old sft data should be done this way as well.
        if "format_version" in example and example['format_version'] is not None: 
            assert(example['format_version'] == 1) # 1 is a dummy value
            prompt = example["prompt"]
            answer = example["answer"]
        else:
            prompt = example["annotated_text"].split("\nAnswer:\n")[0] + "\nAnswer:\n"
            answer = example["annotated_text"].split("\nAnswer:\n")[1]

        # Tokenize separately
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=True,
            max_length=max_length,
            padding=False,
        )["input_ids"]

        answer_tokens = tokenizer(
            answer,
            truncation=True,
            add_special_tokens=False,
            max_length=max_length,
            padding=False,
        )["input_ids"] + [tokenizer.eos_token_id]
        # BUG: manually add eos token to answer tokens

        # Concatenate
        input_ids = prompt_tokens + answer_tokens
        input_ids = input_ids[:max_length]  # Truncate if needed
        non_padding_len = len(input_ids)

        labels = ([-100] * len(prompt_tokens) + answer_tokens)[:max_length]

        # NOW pad to max_length if needed
        padding_length = max_length - len(input_ids)

        input_ids += [tokenizer.pad_token_id] * padding_length
        labels += [-100] * padding_length
        attention_mask = [1] * non_padding_len + [0] * padding_length

        assert len(input_ids) == len(labels) == len(attention_mask) == max_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = train_dataset.map(tokenize_fn, batched=False, fn_kwargs={"max_length" : pretrain_args.max_seq_length })

    keep_keys = ["input_ids", "attention_mask", "labels"]
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in keep_keys])

    if not eval_dataset or all(v is None for v in eval_dataset.values()):
        logger.warning("No valid eval splits found, disabling evaluation.")
        training_args.eval_strategy = "no"
        training_args.do_eval = False
        eval_dataset = None
    else:
        eval_dataset = {key: ds.map(tokenize_fn, batched=False, remove_columns=["annotated_text"], fn_kwargs={"max_length" : pretrain_args.max_seq_length}) for key, ds in eval_dataset.items()}
        eval_dataset = {
            k: ds.remove_columns([col for col in ds.column_names if col not in keep_keys])
            for k, ds in eval_dataset.items()
        }

    if accelerator.is_main_process:
        logger.info(f"Training set size: {train_dataset}")
        logger.info(f"Evaluation set size: {eval_dataset}")

    set_use_special_dblookup_tokens(pretrain_args.use_special_dblookup_tokens)
    set_tokenizer(tokenizer)

    training_args.remove_unused_columns = False

    training_args.compute_loss_func=partial(compute_loss_func, include_eos=True) # pretrain weighted loss
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # data_collator = default_data_collator
    assert tokenizer.eos_token_id in train_dataset[0]["input_ids"], "Eos token not in input_ids"
    check(train_dataset, data_collator, tokenizer)
    # import pdb; pdb.set_trace()

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator, # important to set this to the tokenizer
        compute_loss_func=partial(compute_loss_func, include_eos=True), # include eos token in loss computation, since we want to train the model to generate eos token
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # don't resume training that is the case for tofu ft. only load the model weights
    trainer.train()
    if eval_dataset:
        eval_results = trainer.evaluate()
        logger.info("Evaluation results:\n" + json.dumps(eval_results, indent=4))

    logger.info(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing model to HuggingFace Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, pretrain_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, pretrain_args)
