from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize

from bert.efficient_training import layer_drop


def test__sample_active_layers__start_of_time_schedule__returns_all_layers():
    for seed in [1, 2, 3]:
        torch.random.manual_seed(seed)
        with initialize(config_path="../cramming/config", version_base="1.1"):
            cfg = compose(
                config_name="cfg_pretrain",
                overrides=[
                    "arch.layer_drop.enabled=true",
                    "budget=2",
                    "train.steps=1000",
                    "arch.layer_drop.max_theta=0.5",
                    "arch.num_transformer_layers=10",
                ],
            )
            layers, _ = layer_drop.sample_active_layers(seconds=1, step=1000, cfg=cfg)
            assert set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == set(layers)


def test__sample_active_layers__end_of_time_schedule__correct_number_layers_on_avg():
    torch.random.manual_seed(0)
    with initialize(config_path="../cramming/config", version_base="1.1"):
        cfg = compose(
            config_name="cfg_pretrain",
            overrides=[
                "arch.layer_drop.enabled=true",
                "budget=2",
                "train.steps=1000",
                "arch.layer_drop.max_theta=0.5",
                "arch.num_transformer_layers=10",
            ],
        )
        layers_and_probs = [layer_drop.sample_active_layers(seconds=2 * 60 * 60, step=0, cfg=cfg) for _ in range(1000)]
        lengths = [len(ls) for ls, ps in layers_and_probs]
        # Expected number of layers is (3L-1)/4 of total layers, as in prog layer droping paper.
        error = abs(sum(lengths) / len(lengths) - 7.25)
        assert error < 0.1


def test__sample_active_layers__middle_of_time_schedule__correct_number_layers_on_avg():
    torch.random.manual_seed(0)
    with initialize(config_path="../cramming/config", version_base="1.1"):
        cfg = compose(
            config_name="cfg_pretrain",
            overrides=[
                "arch.layer_drop.enabled=true",
                "budget=2",
                "train.steps=1000",
                "arch.layer_drop.max_theta=0.5",
                "arch.num_transformer_layers=10",
            ],
        )
        layers_and_probs = [layer_drop.sample_active_layers(seconds=1 * 60, step=1000, cfg=cfg) for _ in range(1000)]
        lengths = [len(ls) for ls, ps in layers_and_probs]
        avg_lengths = sum(lengths) / len(lengths)
        assert avg_lengths > 8.0
        assert avg_lengths < 9.5


def test__sample_active_layers__end_of_time_schedule__earlier_layers_more_common_than_later_layers():
    torch.random.manual_seed(0)
    with initialize(config_path="../cramming/config", version_base="1.1"):
        cfg = compose(
            config_name="cfg_pretrain",
            overrides=[
                "arch.layer_drop.enabled=true",
                "budget=2",
                "train.steps=1000",
                "arch.layer_drop.max_theta=0.5",
                "arch.num_transformer_layers=10",
            ],
        )

        layers_and_probs = [layer_drop.sample_active_layers(seconds=2 * 60 * 60, step=0, cfg=cfg) for _ in range(50)]

        layer_counts = defaultdict(lambda: 0)
        for ls, p in layers_and_probs:
            for l in ls:
                layer_counts[l] += 1
        assert layer_counts[9] < layer_counts[0]
        assert layer_counts[8] < layer_counts[1]


def test__sample_active_layers__start_of_step_schedule__returns_all_layers():
    for seed in [1, 2, 3]:
        torch.random.manual_seed(seed)
        with initialize(config_path="../cramming/config", version_base="1.1"):
            cfg = compose(
                config_name="cfg_pretrain",
                overrides=[
                    "arch.layer_drop.enabled=true",
                    "budget=steps",
                    "train.steps=1000",
                    "arch.layer_drop.max_theta=0.5",
                    "arch.num_transformer_layers=10",
                ],
            )
            layers, _ = layer_drop.sample_active_layers(seconds=24 * 60 * 60, step=0, cfg=cfg)
            assert set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == set(layers)


def test__sample_active_layers__end_of_step_schedule__correct_number_layers_on_avg():
    torch.random.manual_seed(0)
    with initialize(config_path="../cramming/config", version_base="1.1"):
        cfg = compose(
            config_name="cfg_pretrain",
            overrides=[
                "arch.layer_drop.enabled=true",
                "budget=steps",
                "train.steps=1000",
                "arch.layer_drop.max_theta=0.5",
                "arch.num_transformer_layers=10",
            ],
        )
        layers_and_probs = [layer_drop.sample_active_layers(seconds=0, step=1000, cfg=cfg) for _ in range(1000)]
        lengths = [len(ls) for ls, ps in layers_and_probs]
        # Expected number of layers is (3L-1)/4 of total layers, as in prog layer droping paper.
        error = abs(sum(lengths) / len(lengths) - 7.25)
        assert error < 0.1


def test__sample_active_layers__middle_of_step_schedule__correct_number_layers_on_avg():
    torch.random.manual_seed(0)
    with initialize(config_path="../cramming/config", version_base="1.1"):
        cfg = compose(
            config_name="cfg_pretrain",
            overrides=[
                "arch.layer_drop.enabled=true",
                "budget=steps",
                "train.steps=1000",
                "arch.layer_drop.max_theta=0.5",
                "arch.num_transformer_layers=10",
            ],
        )
        layers_and_probs = [layer_drop.sample_active_layers(seconds=0, step=10, cfg=cfg) for _ in range(1000)]
        lengths = [len(ls) for ls, ps in layers_and_probs]
        avg_lengths = sum(lengths) / len(lengths)
        assert avg_lengths > 8.0
        assert avg_lengths < 9.5


if __name__ == "__main__":
    # Plot the drop probabilities. A nice check to complement the unit tests.
    torch.random.manual_seed(0)
    mins_list = range(0, 20, 1)
    avg_lengths = []
    for mins in mins_list:
        with initialize(config_path="../cramming/config", version_base="1.1"):
            cfg = compose(
                config_name="cfg_pretrain",
                overrides=[
                    "arch.layer_drop.enabled=true",
                    "budget=2",
                    "train.steps=1000",
                    "arch.layer_drop.max_theta=0.5",
                    "arch.num_transformer_layers=10",
                ],
            )
            layers_and_probs = [layer_drop.sample_active_layers(seconds=mins * 60, step=1000, cfg=cfg) for _ in range(1000)]
            lengths = [len(ls) for ls, ps in layers_and_probs]
            avg_lengths.append(sum(lengths) / len(lengths))

    plt.plot(mins_list, avg_lengths)
    plt.axhline(7.25)
    plt.savefig("lengths.png", dpi=200)
    plt.close()
