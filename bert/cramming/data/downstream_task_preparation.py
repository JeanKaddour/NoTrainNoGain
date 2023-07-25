"""Prepare downstream tasks evaluations."""
import logging
import os
from collections import defaultdict

import datasets
import torch
from datasets import load_dataset

from ..backend.utils import prepare_downstream_dataloader
from .pretraining_preparation import main_process_first

log = logging.getLogger(__name__)


def get_sentences(examples, sentence_keys):
    return tuple(examples[key] for key in sentence_keys if key is not None)


def prepare_task_dataloaders(tokenizer, cfg_eval, cfg_impl):
    """Load all datasets in eval.tasks for finetuning and testing."""
    cfg_eval.path = os.path.expanduser(cfg_eval.path)
    datasets.enable_caching()  # We can cache these
    max_seq_length = cfg_eval.max_seq_length
    tasks = defaultdict(dict)

    for task_name, task_details in cfg_eval.tasks.items():
        log.info(f"Preparing data for task {task_name}.")
        tasks[task_name]["details"] = task_details
        raw_datasets = load_dataset(task_details.collection, task_name, cache_dir=cfg_impl.path)
        if not task_details.regression:
            if "label" in task_details and task_details.label is not None and len(task_details.label) > 0:
                label_list = [task_details.label]
            else:
                label_list = raw_datasets["train"].features["label"].names
            log.info(f"{task_name} has classes {label_list}.")
            tasks[task_name]["num_classes"] = len(label_list)
        else:
            tasks[task_name]["num_classes"] = 1
            label_list = None
        sentence_keys = task_details.structure

        def preprocess_function(examples):
            texts = get_sentences(examples, sentence_keys)
            result = tokenizer(
                *texts,
                max_length=max_seq_length,
                truncation=True,
                pad_to_multiple_of=cfg_impl.pad_to_multiple_of,
            )

            if "label" in examples:
                result["labels"] = examples["label"]
            if task_name == "multirc":
                result["p_idx"] = [ex["paragraph"] for ex in examples["idx"]]
                result["q_idx"] = [ex["question"] for ex in examples["idx"]]
                result["a_idx"] = [ex["answer"] for ex in examples["idx"]]
            return result

        with main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                batch_size=1024,
                load_from_cache_file=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )

        train_dataset = processed_datasets["train"]
        train_dataset.set_format("torch")
        assert cfg_eval.evaluation_set in ["validation", "test"]
        eval_dataset = processed_datasets[f"{cfg_eval.evaluation_set}_matched" if task_name == "mnli" else cfg_eval.evaluation_set]
        eval_dataset.set_format("torch")
        if task_name == "mnli":
            # Extra task loader for MNLI
            extra_eval_dataset = processed_datasets[f"{cfg_eval.evaluation_set}_mismatched"]
            extra_eval_dataset.set_format("torch")
        else:
            extra_eval_dataset = None

        train_dataloader, eval_dataloader, extra_eval_dataloader = _build_dataloaders(
            tokenizer,
            train_dataset,
            eval_dataset,
            extra_eval_dataset,
            cfg_impl,
        )

        tasks[task_name]["trainloader"] = train_dataloader
        tasks[task_name]["validloader"] = eval_dataloader
        tasks[task_name]["extra_validloader"] = extra_eval_dataloader

        # Log overviews so we always know what's going on with weird tokenization tricks
        random_sentence_idx = torch.randint(0, len(train_dataset), (1,)).item()
        input_data = train_dataset[random_sentence_idx]["input_ids"].squeeze()  # squeeze because hf has leading dim

        log.info(f"Random sentence with seq_length {tokenizer.model_max_length} from trainset of size {len(train_dataset):,}: ...")
        log.info(tokenizer.batch_decode(input_data[None])[0])
        log.info("... is tokenized into ...")
        log.info("_".join(tokenizer.decode(t) for t in input_data))
        if label_list is not None:
            log.info(f"Correct Answer: {label_list[train_dataset[random_sentence_idx]['labels']]}")
        else:
            log.info(f"Correct Answer: {train_dataset[random_sentence_idx]['labels']}")
        random_sentence_idx = torch.randint(0, len(eval_dataset), (1,)).item()
        input_data = eval_dataset[random_sentence_idx]["input_ids"].squeeze()  # squeeze because hf has leading dim

        log.info(f"Random sentence from validset of size {len(eval_dataset):,}: ...")
        log.info(tokenizer.batch_decode(input_data[None])[0])
        if label_list is not None:
            log.info(f"Correct Answer: {label_list[eval_dataset[random_sentence_idx]['labels']]}")
        else:
            log.info(f"Correct Answer: {eval_dataset[random_sentence_idx]['labels']}")

    return tasks


def _build_dataloaders(tokenizer, train_dataset, eval_dataset, extra_eval_dataset, cfg_impl):
    """Construct dataloaders according to cfg_impl settings. Validation samplers always repeat on all devices."""
    train_dataloader = prepare_downstream_dataloader(train_dataset, tokenizer, "training", cfg_impl)
    eval_dataloader = prepare_downstream_dataloader(eval_dataset, tokenizer, "eval", cfg_impl)
    if extra_eval_dataset is not None:
        extra_eval_dataloader = prepare_downstream_dataloader(extra_eval_dataset, tokenizer, "eval", cfg_impl)
    else:
        extra_eval_dataloader = None
    return train_dataloader, eval_dataloader, extra_eval_dataloader
