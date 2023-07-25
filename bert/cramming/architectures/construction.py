"""Interface to construct models."""

import logging

from cramming.utils import is_main_process

from .fixed_cramlm import construct_fixed_cramlm
from .funnel_transformers import construct_scriptable_funnel
from .huggingface_interface import construct_huggingface_model
from .recurrent_transformers import construct_scriptable_recurrent
from .sanity_check import SanityCheckforPreTraining
from .scriptable_bert import construct_scriptable_bert

log = logging.getLogger(__name__)


def construct_model(cfg_arch, vocab_size, downstream_classes=None):
    model = None
    if cfg_arch.architectures is not None:
        # attempt to solve locally
        if "ScriptableMaskedLM" in cfg_arch.architectures:
            model = construct_scriptable_bert(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableFunnelLM" in cfg_arch.architectures:
            model = construct_scriptable_funnel(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableRecurrentLM" in cfg_arch.architectures:
            model = construct_scriptable_recurrent(cfg_arch, vocab_size, downstream_classes)
        elif "SanityCheckLM" in cfg_arch.architectures:
            model = SanityCheckforPreTraining(cfg_arch.width, vocab_size)
        elif "FusedCraMLM" in cfg_arch.architectures:
            model = construct_fixed_cramlm(cfg_arch, vocab_size, downstream_classes)

    if model is not None:  # Return local model arch
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with architecture {cfg_arch.architectures[0]} loaded with {num_params:,} parameters.")
        return model

    try:  # else try on HF
        model = construct_huggingface_model(cfg_arch, vocab_size, downstream_classes)
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with config {cfg_arch} loaded with {num_params:,} parameters.")
        return model
    except Exception as e:
        raise ValueError(f"Invalid model architecture {cfg_arch.architectures} given. Error: {e}")
