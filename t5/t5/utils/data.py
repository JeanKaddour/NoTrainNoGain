import datasets
from datasets.iterable_dataset import IterableDataset
from omegaconf import open_dict
from torch.utils.data import DataLoader

from .copied import (
    DataCollatorForNI,
    DataCollatorForT5MLM,
    compute_input_and_target_lengths,
    tokenize_function,
)


def load_dataset_splits(args):
    if args.mode == "pt":
        dataset = datasets.load_dataset(
            "c4",
            "en",
            streaming=True,
        )

        dataset = dataset.remove_columns(["timestamp", "url"])

        dataset_splits = {
            "train": dataset["train"],
            "test": dataset["validation"],
        }

        assert (
            dataset["train"].n_shards == 1024
        ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"
    elif args.mode == "ft":
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task,
        )
    else:
        raise NotImplementedError

    return dataset_splits


def process_dataset(dataset_splits, args, tokenizer):
    if args.mode == "pt":
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )

            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "in_length": before_mask_input_length,
                },
                remove_columns=["text"],
            )

            dataset_split = dataset_split.shuffle(
                seed=args.seed, buffer_size=args.data.shuffle_buffer_size
            )
            final_datasets[split] = dataset_split
    elif args.mode == "ft":
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets


def get_data_collator(tokenizer, config, args):
    if args.mode == "pt":
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length,
            target_length=args.data.target_length,
            pad_token_id=config.pad_token_id,
        )
    elif args.mode == "ft":
        data_collator = DataCollatorForNI(
            tokenizer,
            padding="longest",
            max_source_length=args.data.max_seq_len,
            max_target_length=args.data.max_target_len,
            label_pad_token_id=-100,
            pad_to_multiple_of=1,
            add_task_name=args.data.add_task_name,
            add_task_definition=args.data.add_task_definition,
            num_pos_examples=args.data.num_pos_examples,
            num_neg_examples=args.data.num_neg_examples,
            add_explanation=args.data.add_explanation,
            tk_instruct=args.data.tk_instruct,
        )
    else:
        raise NotImplementedError

    return data_collator


def get_dataloaders(tokenizer, config, args):
    dataset_splits = load_dataset_splits(args)
    dataset = process_dataset(
        dataset_splits=dataset_splits, args=args, tokenizer=tokenizer
    )
    data_collator = get_data_collator(tokenizer=tokenizer, config=config, args=args)

    is_iterable = isinstance(dataset["train"], IterableDataset)

    dataloaders = {}

    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        if split in ["test"]:
            batch_size *= 2

        shuffle = (split == "train") and not is_iterable

        if args.mode == "ft" and split == "train":
            assert shuffle is True
        else:
            assert shuffle is False

        dataloaders[split] = DataLoader(
            dataset[split],
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Add & Check args about data loaders
    with open_dict(args):
        if not is_iterable:
            args.data.train_batches = len(dataloaders["train"])
            args.data.test_batches = len(dataloaders["test"])

        if args.optim.epochs > 0:
            assert not is_iterable
            args.optim.total_steps = (
                len(dataloaders["train"]) // args.optim.grad_acc
            ) * args.optim.epochs

        # We increase eval BS by 2, so decrease number of eval steps
        args.eval.corrected_steps = args.eval.steps / 2

    return dataloaders["train"], dataloaders["test"]
