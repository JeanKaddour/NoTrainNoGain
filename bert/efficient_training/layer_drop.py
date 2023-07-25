import math

import torch


def sample_active_layers(seconds: int, step: int, cfg) -> tuple[list[int], float]:
    total_layers = cfg.arch.num_transformer_layers
    max_drop_prob = _get_drop_prob(seconds, step, cfg)

    active_layers: list[int] = []
    for layer_i in range(0, total_layers):
        layer_drop_prob = max_drop_prob / total_layers * (layer_i + 1)
        if torch.bernoulli(torch.tensor(1.0 - layer_drop_prob)):
            active_layers.append(layer_i)
    return active_layers, max_drop_prob


def _get_drop_prob(seconds: int, step: int, cfg) -> float:
    if cfg.budget == "steps":
        t = step
        T = cfg.train.steps
    else:
        budget_seconds = cfg.budget * 60 * 60
        t = seconds
        T = budget_seconds
    gamma = cfg.arch.layer_drop.gamma_factor / T
    min_theta = cfg.arch.layer_drop.max_theta
    return 1 - (min_theta + (1 - min_theta) * math.exp(-gamma * t))
