import math
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from .t5 import MyT5


def copy_weights_and_states_to_layer_idx(
    model: torch.nn.Module,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
) -> None:
    if isinstance(model, MyT5):
        copy_params_t5(
            model=model,
            next_active_layer_idx=next_active_layer_idx,
            curr_active_layer_idx=curr_active_layer_idx,
        )
    elif isinstance(model, torch._dynamo.eval_frame.OptimizedModule) and isinstance(
        model._orig_mod, MyT5
    ):
        copy_params_t5(
            model=model,
            next_active_layer_idx=next_active_layer_idx,
            curr_active_layer_idx=curr_active_layer_idx,
        )
    else:
        raise NotImplementedError(
            f"Copying weights and states to layer index is not implemented for model {model.__class__.__name__}"
        )


def _copy_params_(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
) -> None:
    for source_param, target_param in zip(
        source_model.parameters(), target_model.parameters()
    ):
        target_param.data.copy_(source_param.data)


def copy_params_t5(
    model: torch.nn.Module,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
) -> None:
    if curr_active_layer_idx is None:
        curr_active_layer_idx = next_active_layer_idx - 1

    _copy_params_(
        model.encoder.block[curr_active_layer_idx],
        model.encoder.block[next_active_layer_idx],
    )

    _copy_params_(
        model.decoder.block[curr_active_layer_idx],
        model.decoder.block[next_active_layer_idx],
    )


class StackingScheduler(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_active_layers: int,
        num_train_steps: int = 1000,
        step_counter: float = 0.0,
        num_layers_to_add: int = 4,
    ):
        self.model = model
        self.model_optimizer = optimizer
        self.num_active_layers = num_active_layers
        self.initial_num_active_layers = num_active_layers
        self.num_train_steps = num_train_steps
        self.step_counter = step_counter
        self.num_layers_to_add = num_layers_to_add

    def set_active_layers(self, num_active_layers: int) -> None:
        self.model.set_active_layers(num_active_layers)

    @abstractmethod
    def step(self, step: Optional[float] = None) -> dict:
        pass


class NoOpStackingScheduler(StackingScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, step: Optional[float] = None) -> dict:
        pass


class DropScheduler(StackingScheduler):
    def __init__(
        self,
        seconds_budget: Optional[int] = None,
        num_layers_to_add: int = 12,
        gamma_factor: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seconds_budget = seconds_budget
        self.gamma_factor = gamma_factor
        self.max_theta = 0.5
        self.total_layers = num_layers_to_add

    def step(self, seconds_counter):
        active_layers = self.sample_active_layers(seconds_counter)
        self.model.set_active_layers(
            num_active_layers=None,
            layer_ids=active_layers,
        )

    def sample_active_layers(self, seconds_counter):
        avg_drop_prob = self._get_drop_prob(seconds_counter)

        active_layers: list[int] = []
        for layer_i in range(1, self.total_layers + 1):
            layer_drop_prob = avg_drop_prob / self.total_layers * layer_i
            if torch.bernoulli(torch.tensor(1.0 - layer_drop_prob)):
                active_layers.append(layer_i - 1)

        return active_layers

    def _get_drop_prob(self, seconds) -> float:
        gamma = self.gamma_factor / self.seconds_budget
        min_theta = self.max_theta
        return 1 - (min_theta + (1 - min_theta) * math.exp(-gamma * seconds))


class ManualStackingScheduler(StackingScheduler):
    def __init__(
        self,
        seconds_budget: Optional[int] = None,
        step_fractions: Optional[List[float]] = [0.25, 0.5, 0.75],
        doubling: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            step_fractions is not None
        ), "step_fractions must be provided for manual stacking"
        self.steps = [step * self.num_train_steps for step in step_fractions]

        print(f"Stacking Steps: {self.steps}, len: {len(self.steps)}")
        self.set_active_layers(self.num_active_layers)
        self.seconds_budget = seconds_budget
        self.doubling = doubling

    def _get_fake_step(self, seconds_counter: float) -> float:
        if seconds_counter == 0.0:
            return 0.0
        else:
            return seconds_counter / self.seconds_budget * self.num_train_steps

    def update_layers(self):
        if self.doubling:
            for i in range(self.num_active_layers, self.num_active_layers * 2):
                copy_weights_and_states_to_layer_idx(
                    model=self.model,
                    next_active_layer_idx=i,
                    curr_active_layer_idx=i - self.num_active_layers,
                )

            self.num_active_layers = self.num_active_layers * 2
        else:
            copy_weights_and_states_to_layer_idx(
                model=self.model,
                next_active_layer_idx=self.num_active_layers,
            )
            self.num_active_layers += 1

        self.set_active_layers(self.num_active_layers)

    def step(self, seconds_counter):
        assert self.seconds_budget is not None
        self.step_counter = self._get_fake_step(seconds_counter)

        if len(self.steps) > 0 and self.step_counter >= self.steps[0]:
            self.steps.pop(0)
            self.update_layers()

        return {"num_active_layers": self.num_active_layers}


def get_stacking_scheduler(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_active_layers: int,
    num_layers_to_add: int,
    num_train_steps: int,
    typ: bool,
    step_fractions: List[float] = None,
    doubling: bool = False,
    seconds_budget: Optional[int] = None,
    gamma_factor: int = None,
) -> StackingScheduler:
    if typ == "none":
        return NoOpStackingScheduler(
            model=model,
            optimizer=optimizer,
            num_active_layers=num_active_layers,
            num_layers_to_add=num_layers_to_add,
        )
    elif typ == "stack":
        return ManualStackingScheduler(
            model=model,
            optimizer=optimizer,
            num_active_layers=num_active_layers,
            num_train_steps=num_train_steps,
            num_layers_to_add=num_layers_to_add,
            step_fractions=step_fractions,
            doubling=doubling,
            seconds_budget=seconds_budget,
        )
    elif typ == "drop":
        return DropScheduler(
            model=model,
            optimizer=optimizer,
            num_active_layers=num_active_layers,
            num_layers_to_add=num_layers_to_add,
            seconds_budget=seconds_budget,
            gamma_factor=gamma_factor,
        )
