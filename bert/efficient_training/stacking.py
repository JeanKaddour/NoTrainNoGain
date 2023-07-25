import math
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from cramming.architectures.gpt2 import GPT2LMHeadModel
from cramming.architectures.scriptable_bert import ScriptableLMForPreTraining
from cramming.architectures.t5 import MyT5

StackingType = Literal["manual", "adaptive"]


def _copy_optimizer_state(state: dict) -> dict:
    new_state = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            new_state[key] = value.clone().detach()
            new_state[key].requires_grad = False
        else:
            new_state[key] = value
    return new_state


def copy_weights_and_states_to_layer_idx(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
    copy_optim_states: bool = True,
    reset_optim_states: bool = False,
) -> None:
    if isinstance(model, ScriptableLMForPreTraining):
        copy_params_bert(model, next_active_layer_idx, curr_active_layer_idx)
        if copy_optim_states:
            copy_optim_states_bert(
                optimizer,
                model,
                next_active_layer_idx,
                curr_active_layer_idx,
            )
    elif isinstance(model, GPT2LMHeadModel):
        copy_params_gpt(model, next_active_layer_idx, curr_active_layer_idx)
        if copy_optim_states:
            copy_optim_states_gpt(
                optimizer,
                model,
                next_active_layer_idx,
                curr_active_layer_idx,
            )
    elif isinstance(model, MyT5) or (isinstance(model, torch._dynamo.eval_frame.OptimizedModule) and isinstance(model._orig_mod, MyT5)):
        torch._dynamo.reset()
        copy_params_t5(model, next_active_layer_idx, curr_active_layer_idx)
        if copy_optim_states:
            copy_optim_states_t5(optimizer, model, next_active_layer_idx, curr_active_layer_idx)
    else:
        raise NotImplementedError(f"Copying weights and states to layer index is not implemented for model {model.__class__.__name__}")
    if reset_optim_states:
        optimizer.state = {}


def freeze_params_up_to_layer(model: torch.nn.Module, layer_idx: int, layer_type: str) -> None:
    if layer_type == "scriptable_bert":
        layers = model.encoder.layers
    elif layer_type == "gpt":
        layers = model.transformer.h
    elif layer_type == "t5_encoder":
        layers = model.encoder.block
    elif layer_type == "t5_decoder":
        layers = model.decoder.block
    else:
        raise ValueError("Invalid layer_type")

    for i in range(layer_idx):
        for param in layers[i].parameters():
            param.requires_grad = False


def freeze_weights_up_to_layer_idx(model: torch.nn.Module, layer_idx: int) -> None:
    if isinstance(model, ScriptableLMForPreTraining):
        freeze_params_up_to_layer(model, layer_idx, "scriptable_bert")
    elif isinstance(model, GPT2LMHeadModel):
        freeze_params_up_to_layer(model, layer_idx, "gpt")
    elif isinstance(model, MyT5):
        freeze_params_up_to_layer(model, layer_idx, "t5_encoder")
        freeze_params_up_to_layer(model, layer_idx, "t5_decoder")
    else:
        raise NotImplementedError(f"Freezing weights up to layer index is not implemented for model {model.__class__.__name__}")


def _copy_params_(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
) -> None:
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_(source_param.data)


#### T5 ####


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


def copy_optim_states_t5(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
) -> None:
    if curr_active_layer_idx is None:
        curr_active_layer_idx = next_active_layer_idx - 1
    source_block_encoder = model.encoder.block[curr_active_layer_idx]
    target_block_encoder = model.encoder.block[next_active_layer_idx]

    source_block_decoder = model.decoder.block[curr_active_layer_idx]
    target_block_decoder = model.decoder.block[next_active_layer_idx]

    source_params = (
        *source_block_encoder.parameters(),
        *source_block_decoder.parameters(),
    )
    target_params = (
        *target_block_encoder.parameters(),
        *target_block_decoder.parameters(),
    )

    for source_param, target_param in zip(source_params, target_params):
        assert source_param in optimizer.state, f"Source param {source_param} not in optimizer state"
        optimizer.state[target_param] = _copy_optimizer_state(optimizer.state[source_param])


#### GPT2 ####


def copy_params_gpt(
    model: torch.nn.Module,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
) -> None:
    if curr_active_layer_idx is None:
        curr_active_layer_idx = next_active_layer_idx - 1
    _copy_params_(
        model.transformer.h[curr_active_layer_idx],
        model.transformer.h[next_active_layer_idx],
    )
    model.transformer.h[next_active_layer_idx].attn.layer_index = next_active_layer_idx


def copy_optim_states_gpt(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
) -> None:
    if curr_active_layer_idx is None:
        curr_active_layer_idx = next_active_layer_idx - 1
    source_block = model.transformer.h[curr_active_layer_idx]
    target_block = model.transformer.h[next_active_layer_idx]
    for source_param, target_param in zip(source_block.parameters(), target_block.parameters()):
        assert source_param in optimizer.state, f"Source param {source_param} not in optimizer state"
        optimizer.state[target_param] = _copy_optimizer_state(optimizer.state[source_param])


#### BERT ####


def _clone_layer_params(model: torch.nn.Module, layer_idx: int) -> List[torch.nn.Parameter]:
    layer = model.encoder.layers[layer_idx]
    return [p.clone() for p in layer.parameters()]


def paste_layer_params(model: torch.nn.Module, layer_idx: int, params: List[torch.nn.Parameter]) -> None:
    layer = model.encoder.layers[layer_idx]
    for p_stacked, p in zip(layer.parameters(), params):
        p_stacked.data = p.data


def copy_params_bert(
    model: torch.nn.Module,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
) -> None:
    if curr_active_layer_idx is None:
        curr_active_layer_idx = next_active_layer_idx - 1
    _copy_params_(
        model.encoder.layers[curr_active_layer_idx],
        model.encoder.layers[next_active_layer_idx],
    )


def copy_optim_states_bert(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    next_active_layer_idx: int,
    curr_active_layer_idx: Optional[int] = None,
) -> None:
    if curr_active_layer_idx is None:
        curr_active_layer_idx = next_active_layer_idx - 1
    source_block = model.encoder.layers[curr_active_layer_idx]
    target_block = model.encoder.layers[next_active_layer_idx]
    for source_param, target_param in zip(source_block.parameters(), target_block.parameters()):
        assert source_param in optimizer.state, f"Source param {source_param} not in optimizer state"
        optimizer.state[target_param] = _copy_optimizer_state(optimizer.state[source_param])


#### Stacking ####


class StackingScheduler(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_active_layers: int,
        T_max: int = 1000,
        num_train_steps: int = 1000,
        step_counter: float = 0.0,
        start_steps: int = 1000,
        num_layers_to_add: int = 4,
        copy_optim_states: bool = True,
    ):
        self.model = model
        self.model_optimizer = optimizer
        self.num_active_layers = num_active_layers
        self.initial_num_active_layers = num_active_layers
        self.T_max = T_max
        self.num_train_steps = num_train_steps
        self.step_counter = step_counter
        self.start_steps = start_steps
        self.num_layers_to_add = num_layers_to_add
        self.copy_optim_states = copy_optim_states

    def set_active_layers(self, num_active_layers: int) -> None:
        self.model.set_active_layers(num_active_layers)

    @abstractmethod
    def step(self, step: Optional[float] = None) -> None:
        pass


class NoOpStackingScheduler(StackingScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, step: Optional[float] = None) -> None:
        pass


ManualStackingFunction = Literal["monomial", "cosine", "manual"]


class ManualStackingScheduler(StackingScheduler):
    def __init__(
        self,
        function: ManualStackingFunction,
        balance_factor: float = 1.2,
        freeze_bottom_layers: bool = False,
        adjust_lr: bool = False,
        warmup_steps: int = 1000,
        seconds_budget: Optional[int] = None,
        step_fractions: Optional[List[float]] = [0.25, 0.5, 0.75],
        doubling: bool = False,
        doubling_interpolation: bool = False,
        reset_optim: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.balance_factor = balance_factor
        if function == "monomial":
            self.steps = self._compute_steps_monomial()
        elif function == "cosine":
            self.steps = self._compute_steps_cosine()
        elif function == "manual":
            assert step_fractions is not None, "step_fractions must be provided for manual stacking"
            self.steps = [step * self.num_train_steps for step in step_fractions]
        print(f"Stacking Steps: {self.steps}, len: {len(self.steps)}")
        self.freeze_bottom_layers = freeze_bottom_layers
        self.set_active_layers(self.num_active_layers)
        self.adjust_lr = adjust_lr
        self.warmup_steps = warmup_steps
        self.seconds_budget = seconds_budget
        self.doubling = doubling
        self.doubling_interpolation = doubling_interpolation
        self.reset_optimizer = reset_optim
        self._adjust_lr()

    def _compute_steps_monomial(self):
        percentiles = [i / (self.num_layers_to_add) for i in range(self.num_layers_to_add + 1)]
        intervals = [((self.T_max - self.start_steps) * percentile**self.balance_factor) for percentile in percentiles]
        steps = [int(self.start_steps + interval) for interval in intervals]
        return steps[1:]

    def _compute_steps_cosine(self):
        ys = np.linspace(0.0, 2.0, num=self.num_layers_to_add + 1)
        # We have the curve y = - cos(x), and want the range between 0 and pi/2.
        xs = np.arccos(1 - ys)
        duration = self.T_max - self.start_steps
        steps_real = xs / np.pi * duration + self.start_steps
        return np.round(steps_real).astype(int).tolist()[1:]

    def _get_fake_step(self, step: float) -> float:
        if step == 0.0:
            return 0.0
        else:
            return step / self.seconds_budget * self.num_train_steps

    def update_layers(self):
        self._adjust_lr(inverse=True)

        if self.doubling:
            if self.doubling_interpolation:
                cloned_weights = {i: _clone_layer_params(self.model, i) for i in range(self.num_active_layers)}
                # interweave the cloned weights with the original weights
                # A, A', B, B', C, C', D, D'
                # A, B,  C, D,  E, F,  G, H
                for i in range(0, self.num_active_layers * 2, 2):
                    paste_layer_params(self.model, i, cloned_weights[i // 2])
                    paste_layer_params(self.model, i + 1, cloned_weights[i // 2])
            else:
                for i in range(self.num_active_layers, self.num_active_layers * 2):
                    copy_weights_and_states_to_layer_idx(
                        self.model,
                        self.model_optimizer,
                        next_active_layer_idx=i,
                        curr_active_layer_idx=i - self.num_active_layers,
                        copy_optim_states=self.copy_optim_states,
                        reset_optim_states=self.reset_optimizer,
                    )
            self.num_active_layers = self.num_active_layers * 2
        else:
            copy_weights_and_states_to_layer_idx(
                self.model,
                self.model_optimizer,
                next_active_layer_idx=self.num_active_layers,
                copy_optim_states=self.copy_optim_states,
                reset_optim_states=self.reset_optimizer,
            )
            self.num_active_layers += 1
        self.set_active_layers(self.num_active_layers)
        self._adjust_lr(inverse=False)

    def step(self, step: Optional[float] = None):
        self.step_counter = self.step_counter + 1.0 if not self.seconds_budget else self._get_fake_step(step)
        if self.step_counter == self.warmup_steps:
            self._adjust_lr(inverse=False)
        if len(self.steps) > 0 and self.step_counter >= self.steps[0]:
            self.steps.pop(0)
            self.update_layers()

            if self.freeze_bottom_layers:
                freeze_weights_up_to_layer_idx(self.model, self.num_active_layers)
                print(f"Freezing up to layer {self.num_active_layers}")
        if self.step_counter == self.T_max:
            self._adjust_lr(inverse=True)

    def _adjust_lr(self, inverse: bool = False, factor_transform: str = "x^1/4"):
        if self.adjust_lr:
            old_lr = self.model_optimizer.param_groups[0]["lr"]
            factor = self.num_active_layers / (self.initial_num_active_layers + self.num_layers_to_add)
            if factor_transform == "sqrt":
                factor = math.sqrt(factor)
            elif factor_transform == "x^1/3":
                factor = factor ** (1 / 3)
            elif factor_transform == "x^1/4":
                factor = factor ** (1 / 4)
            elif factor_transform == "linear":
                pass
            else:
                raise ValueError(f"Unknown factor transform: {factor_transform}")

            new_lr = old_lr * factor if inverse else old_lr * (1 / factor)
            self.model_optimizer.param_groups[0]["lr"] = new_lr


class AdaptiveStackingScheduler(StackingScheduler):
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        freq: int = 100,
        theta_transform: Callable = torch.sigmoid,
        num_samples: int = 10,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.freq = freq
        self.theta_transform = theta_transform
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.theta_optimizer: Optional[torch.optim.Optimizer] = None
        self.theta: Optional[torch.nn.parameter.Parameter] = None
        self.reset_theta()
        self.set_active_layers(self.num_active_layers)

    def theta_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        ps = self.theta_transform(self.theta)
        gs = torch.bernoulli(torch.ones(self.num_samples) * ps)
        data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset,
                replacement=False,
                generator=torch.Generator().manual_seed(random.randint(0, 2**32)),
            ),
            pin_memory=False,
        )
        device = next(self.model.parameters()).device
        train_losses = []
        for i, batch in enumerate(data_loader):
            if i == self.num_samples:
                break
            tmp_num_active_layers = self.num_active_layers if gs[i] == 0 else self.num_active_layers + 3
            self.set_active_layers(tmp_num_active_layers)
            batch = batch.to(device)
            train_losses.append(self.model(batch).detach())
        baseline = torch.mean(torch.stack(train_losses))
        loss = torch.tensor([0.0], device=device)
        for j in range(self.num_samples):
            loss += (gs[j] * torch.log(ps) + (1.0 - gs[j]) * torch.log(1 - ps)) * (train_losses[j] - baseline)
        loss /= self.num_samples - 1
        loss.backward()
        return loss.item(), baseline.item()

    def reset_theta(self):
        self.theta = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.theta_optimizer = torch.optim.SGD([self.theta], lr=1.0)

    def step(self) -> Optional[dict]:
        self.step_counter += 1
        if self.step_counter % self.freq == 0 and self.num_layers_to_add > 0:
            copy_weights_and_states_to_layer_idx(
                self.model,
                self.model_optimizer,
                next_active_layer_idx=self.num_active_layers,
                curr_active_layer_idx=self.num_active_layers - 1,
            )
            theta_loss, baseline = self.theta_loss()
            # todo(jean): eventually, we should clean this up but for rapid prototyping, might be useful
            print(f"before optim step: theta: {self.theta.item()}, prob: {self.theta_transform(self.theta).item()}")
            self.theta_optimizer.step()
            print(f"post optim step: theta: {self.theta.item()}, prob: {self.theta_transform(self.theta).item()}")
            self.theta_optimizer.zero_grad()
            is_new_layer_ready = torch.bernoulli(self.theta_transform(self.theta))
            log_message = ""
            if is_new_layer_ready:
                self.num_active_layers += 1
                self.num_layers_to_add -= 1
                self.set_active_layers(self.num_active_layers)
                self.reset_theta()
                log_message += "added layer! \t"
            else:
                log_message += "not added layer! \t"
            log_message += f"now: num_layers_to_add: {self.num_layers_to_add} active layers: {self.num_active_layers}"
            print(log_message)
            return {
                "ada_stack_baseline": baseline,
                "ada_stack_theta": self.theta.item(),
                "ada_stack_prob": self.theta_transform(self.theta).item(),
                "ada_stack_loss": theta_loss,
                "num_active_layers": self.num_active_layers,
            }


def get_stacking_scheduler(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_active_layers: int,
    num_layers_to_add: int,
    start_steps: int,
    T_max: int,
    num_train_steps: int,
    enabled: bool,
    train_dataset: Optional[Dataset],
    batch_size: int,
    adaptive_freq: int = 100,
    adaptive_num_samples: int = 10,
    manual_function: ManualStackingFunction = "monomial",
    manual_balance_factor: float = 1.0,
    stacking_scheduler_type: StackingType = "manual",
    step_fractions: Optional[list[float]] = None,
    copy_optim_states: bool = False,
    freeze_bottom_layers: bool = False,
    adjust_lr: bool = False,
    doubling: bool = False,
    doubling_interpolation: bool = False,
    warmup_steps: int = 0,
    seconds_budget: Optional[int] = None,
    reset_optim: bool = False,
) -> StackingScheduler:
    if enabled is False:
        # no-op scheduler
        return NoOpStackingScheduler(
            model=model,
            optimizer=optimizer,
            num_active_layers=num_active_layers,
            T_max=T_max,
            start_steps=start_steps,
            num_layers_to_add=num_layers_to_add,
        )
    elif stacking_scheduler_type == "adaptive":
        return AdaptiveStackingScheduler(
            model=model,
            optimizer=optimizer,
            num_active_layers=num_active_layers,
            T_max=T_max,
            start_steps=start_steps,
            num_layers_to_add=num_layers_to_add,
            train_dataset=train_dataset,
            batch_size=batch_size,
            freq=adaptive_freq,
            num_samples=adaptive_num_samples,
        )
    elif stacking_scheduler_type == "manual":
        return ManualStackingScheduler(
            model=model,
            optimizer=optimizer,
            num_active_layers=num_active_layers,
            T_max=T_max,
            num_train_steps=num_train_steps,
            start_steps=start_steps,
            num_layers_to_add=num_layers_to_add,
            balance_factor=manual_balance_factor,
            step_fractions=step_fractions,
            copy_optim_states=copy_optim_states,
            function=manual_function,
            doubling=doubling,
            doubling_interpolation=doubling_interpolation,
            freeze_bottom_layers=freeze_bottom_layers,
            adjust_lr=adjust_lr,
            warmup_steps=warmup_steps,
            seconds_budget=seconds_budget,
            reset_optim=reset_optim,
        )
    else:
        raise ValueError(f"Stacking Scheduler Type {stacking_scheduler_type} is not supported")
