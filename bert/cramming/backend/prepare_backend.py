"""Instantiate backend objects in a congruent format."""
import torch

from .deepspeed_integration import initialize_deepspeed
from .torch_default import initialize_torch

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)


def load_backend(model, dataset, validation_set, tokenizer, cfg_train, cfg_impl, setup=_default_setup):
    if cfg_impl.name == "torch-default":
        return initialize_torch(model, dataset, validation_set, tokenizer, cfg_train, cfg_impl, setup=setup)
    elif cfg_impl.name == "deepspeed":
        return initialize_deepspeed(model, dataset, validation_set, tokenizer, cfg_train, cfg_impl, setup=setup)
    else:
        raise ValueError(f"Invalid backend {cfg_impl.name} given.")
