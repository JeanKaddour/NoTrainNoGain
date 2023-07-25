import torch
from omegaconf import open_dict
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

from .t5 import MyT5


def get_model(args, config):
    klass = {
        "t5": T5ForConditionalGeneration,
        "my_t5": MyT5,
    }[args.model.klass]

    if args.model.checkpoint_path:
        model = klass(config)
        model.load_state_dict(torch.load(args.model.checkpoint_path))
    elif args.model.random_init:
        model = klass(config)
    else:
        model = klass.from_pretrained(
            args.model.name,
            config=config,
        )

    with open_dict(args):
        args.n_all_param = sum([p.nelement() for p in model.parameters()])

    return model


def get_config(args):
    config = AutoConfig.from_pretrained(
        args.model.name,
    )

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    return config


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model.name, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    return tokenizer
