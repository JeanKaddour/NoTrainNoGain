"""Script for a pretraining run using RHO-Loss."""

import datetime
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Optional

import hydra
import torch
from torch import Tensor

import cramming
import cramming.utils
from cramming.utils import validate
from efficient_training import extract_il_losses
from rst.saved_rsts import Task, get_time_per_step

log = logging.getLogger(__name__)


class RhoLossWrapper:
    def __init__(
        self,
        train_batch_size: int,
        mega_batch_size: int,
        micro_batch_size: int,
        il_losses: dict[int, float],
        device: torch.device,
        model: torch.nn.Module,
        task: Task = "bert",
        amp_settings: Optional[dict] = None,
    ):
        self.accumulated_samples: int = 0
        self.example_storage: dict[int, torch.Tensor] = dict()
        self.mega_batch_size: int = mega_batch_size
        self.train_batch_size: int = train_batch_size
        self.micro_batch_size: int = micro_batch_size
        self.il_losses: dict[int, float] = il_losses
        self.rho_losses: list[tuple[int, float]] = list()
        self.task: Task = task
        self.device: torch.device = device
        self.model: torch.nn.Module = model
        self.amp_settings = amp_settings

    def _get_train_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.task == "bert":
            with torch.no_grad():
                with torch.autocast(**self.amp_settings):
                    return self.model.forward_all_losses(**batch)
        else:
            raise NotImplementedError(f"Task {self.task} not implemented.")

    def step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> Optional[dict[str, Tensor]]:
        micro_batches = None
        train_losses = self._get_train_loss(batch)
        il_losses = torch.tensor(
            [self.il_losses[example_id] for example_id in batch["example_ids"]], dtype=train_losses.dtype, device=train_losses.device
        )
        batch_rho_losses = train_losses - il_losses
        self.rho_losses.extend([(i, l.item()) for i, l in zip(batch["example_ids"], batch_rho_losses)])
        self.accumulated_samples += len(batch["example_ids"])
        self.example_storage.update({i: self._retrieve_single_example_and_move_to_CPU(batch, i) for i in batch["example_ids"]})
        if self.accumulated_samples >= self.mega_batch_size:
            train_batch = self._get_top_k_samples(self.example_storage, self.rho_losses, k=self.train_batch_size)
            micro_batches = self.create_list_of_batch_dicts(train_batch)
            self.accumulated_samples = 0
            self.example_storage = {}
            self.rho_losses = []
        return micro_batches

    @staticmethod
    def _retrieve_single_example_and_move_to_CPU(batch: dict[str, torch.Tensor], example_id: int) -> dict[str, torch.Tensor]:
        """Get single example from a mega batch."""
        example_index = batch["example_ids"].index(example_id)
        return {k: v[example_index].cpu() for k, v in batch.items() if k != "example_ids"}

    def create_list_of_batch_dicts(self, examples: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Create a list of batch dicts from a list of examples of size self.microbatch_size"""
        micro_batches = list()
        for i in range(0, len(examples), self.micro_batch_size):
            micro_batches.append(
                {k: torch.stack([example[k] for example in examples[i : i + self.micro_batch_size]], dim=0) for k in examples[0].keys()}
            )
        return micro_batches

    def _get_top_k_samples(self, example_storage: dict[int, torch.Tensor], example_losses: list[tuple[int, float]], k: int = 1) -> list:
        """Get top k samples from a mega batch."""
        example_losses.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = [x[0] for x in example_losses[:k]]
        return [example_storage[i] for i in top_k_ids]


def load_chunks(path: str) -> dict:
    print(f"Loading chunks from {path}")
    final_data = {}
    files = os.listdir(path)
    # filter out directories
    files = [file for file in files if os.path.isfile(os.path.join(path, file))]
    for file in files:
        with open(os.path.join(path, file), "rb") as file:
            chunk_data = pickle.load(file)
            final_data.update(chunk_data)
    return final_data


def main_training_process(cfg, setup):
    """This function controls the central training loop."""
    local_time = time.time()
    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
    dataset, validation_set, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl, cfg.train)
    model_engine, _, _, dataloader, validation_dataloader = cramming.load_backend(
        model,
        dataset,
        validation_set,
        tokenizer,
        cfg.train,
        cfg.impl,
        setup=setup,
    )
    model_engine.train(cfg.train.pretrain_in_train_mode)
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time()
    train_time = time.time()  # Crude time measurement for print_loss_every_nth_step
    seconds_counter = 0.0
    validations_counter = 1
    num_active_layers: list[int] = []
    training_allowed = True
    loss, loss_vals = None, []

    seconds_budget = 60 * 60 * cfg.train.budget  # 1 day by default

    iterable_data = enumerate(dataloader)
    if cfg.train.gradinit.enabled:
        model_engine.gradinit(iterable_data, cfg.train.optim, cfg.train.gradinit)

    il_losses = load_chunks(cfg.train.rho_loss.il_losses_path)

    # max key is the number of examples in the dataset
    num_examples = max(il_losses.keys()) + 1

    rho_loss_wrapper = RhoLossWrapper(
        train_batch_size=cfg.train.batch_size,
        mega_batch_size=cfg.train.rho_loss.mega_batch_size,
        micro_batch_size=cfg.impl.microbatch_size,
        il_losses=il_losses,
        device=model_engine.device,
        model=model,
        task="bert",
        amp_settings=model_engine.amp_settings,
    )
    examples_counter = 0
    # Launch training
    for step, batch in iterable_data:
        step_seconds_counter = seconds_counter if "seconds" in cfg.train.scheduler else None
        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)
        device_batch["example_ids"] = extract_il_losses.get_example_ids_from_batch(examples_counter, len(batch["input_ids"]), num_examples)
        examples_counter += len(batch["input_ids"])
        micro_batches = rho_loss_wrapper.step(device_batch)
        num_active_layers.append(model_engine.get_num_active_layers())
        if cfg.train.track_forward_pass_only:
            seconds_counter += get_time_per_step(
                cfg.impl.microbatch_size, model_engine.get_num_active_layers(), forward_only=True, microbatch_size=cfg.impl.microbatch_size
            )
        if micro_batches is not None:
            for microbatch in micro_batches:
                device_batch = model_engine.to_device(microbatch)
                loss = model_engine.step(device_batch, step_seconds_counter, None)
                loss_vals.append(loss.detach())
                seconds_counter += get_time_per_step(
                    model_engine.current_batch_size,
                    model_engine.get_num_active_layers(),
                    forward_only=False,
                    microbatch_size=cfg.impl.microbatch_size,
                )
                if seconds_counter >= seconds_budget:
                    break

        # Check stopping criteria
        # if check_deadline(wallclock_timer, cfg.budget) or step == cfg.train.steps:
        if seconds_counter >= seconds_budget:
            training_allowed = False
            log.info("Reached deadline. Stopping training ...")

        if (
            cfg.train.validation_set.enabled
            and cfg.impl.validate_every_hours > 0
            and validation_dataloader is not None
            # Validate if we have trained for the given number of hours, or are about to terminate.
            and (seconds_counter > validations_counter * cfg.impl.validate_every_hours * 3600 or not training_allowed)
        ):
            stats["validation_loss"] += [validate(model_engine, validation_dataloader, model_engine.setup["device"])]
            validations_counter += 1
            loss_vals, train_time = collect_stats(
                step,
                seconds_counter,
                loss_vals,
                num_active_layers,
                train_time,
                stats,
                model_engine,
                dataloader,
                cfg,
            )
            log.info(f"Validation loss: {stats['validation_loss'][-1]:2.4f}.")
            state = dict(step=step, tokenizer_name=tokenizer.name_or_path)
            checkpoint_id = f"hours={seconds_counter / 3600:2.2f}"
            if cramming.utils.is_main_process():
                model_engine.save_training_checkpoint(checkpoint_id, state=state)
        # Collect stats and print to console and upload to wandb
        elif step % cfg.impl.print_loss_every_nth_step == 0:
            loss_vals, train_time = collect_stats(
                step,
                seconds_counter,
                loss_vals,
                num_active_layers,
                train_time,
                stats,
                model_engine,
                dataloader,
                cfg,
            )
            if len(stats["loss"]) > 0 and check_early_termination(wallclock_timer, stats["loss"][-1], cfg.impl.early_termination):
                training_allowed = False
                log.info("Loss higher than allowed threshold. Stopping training early...")

        # Checkpointing is triggered from stopping criteria and normal intervals
        if cfg.impl.save_intermediate_checkpoints and step > 0 and step % cfg.impl.save_every_nth_step == 0:
            state = dict(step=step, tokenizer_name=tokenizer.name_or_path)
            checkpoint_id = loss.item()
            if cramming.utils.is_main_process():
                model_engine.save_training_checkpoint(checkpoint_id, state=state)

        if loss is not None and not loss.detach().isfinite():
            training_allowed = False
            log.info("Ending training due to non-finite loss.")
            raise ValueError("Non-finite loss")

        flag_communication(training_allowed)

        if (cfg.dryrun and step > 2) or not training_allowed:
            break

    if cramming.utils.is_main_process():
        # Save to summary:
        metrics = dict(num_params=sum([p.numel() for p in model.parameters()]))
        cramming.utils.save_summary("pretrain", cfg, metrics, stats, time.time() - local_time, setup)
        # Save final checkpoint:
        now = datetime.datetime.now()
        checkpoint_id = f"{''.join(cfg.arch.architectures)}_{now.strftime('%Y-%m-%d')}_{loss:2.4f}"
        model_engine.save_final_model(os.path.join(cfg.base_dir, cfg.name), checkpoint_id, tokenizer, cfg.arch, cfg.dryrun)


def check_deadline(launch_time, hour_limit):
    """These measurements are deliberately wall-clock based."""
    current_time = time.time()
    return True if (current_time - launch_time) / 3600 > hour_limit else False


def check_early_termination(launch_time, loss, early_termination):
    """Early termination based on terrible loss."""
    if early_termination.enabled and loss > early_termination.loss_threshold:
        current_time = time.time()
        return True if (current_time - launch_time) / 3600 > early_termination.budget else False
    else:
        return False


def collect_stats(
    step,
    seconds_counter,
    loss_vals,
    num_active_layers: list[int],
    train_time,
    stats,
    model_engine,
    dataloader,
    cfg,
):
    stats["step"] += [step]
    stats["seconds"] += [seconds_counter]
    stats["hours"] += [seconds_counter / 3600]
    stats["epoch"] += [dataloader.epoch_counter]

    tokens_per_step = cramming.utils.num_processes() * model_engine.record_tokens_per_step()
    stats["tokens"] += [step * tokens_per_step]
    current_lr = model_engine.optimizer.param_groups[0]["lr"]
    log_msg = ""
    if len(loss_vals) > 0:
        stats["loss"] += [torch.stack(loss_vals).mean().item()]  # Averaged loss
        log_msg = f"Train loss {loss_vals[-1].item():2.4f} at step {step} with lr {current_lr:.5f}. "
        log_msg += f"[Avg: {stats['loss'][-1]:2.4f}] "
    if step > 0:
        stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
        estimated_train_finish = str(datetime.timedelta(seconds=stats["train_time"][-1] * cfg.train.steps))
        tokens_per_second = tokens_per_step / stats["train_time"][-1]
        stats["tok/sec"] += [int(tokens_per_second)]
        log_msg += f" Perf: {stats['train_time'][-1]:2.4f}s per step ({tokens_per_second:.0f}t/s). "
        log_msg += f"Estimated Total Train: {estimated_train_finish}."

    # Adaptive optim stats
    stats["lr"] += [current_lr]
    stats["batch_size"] += [model_engine.record_batch_size()]
    stats["seq_length"] = [model_engine.current_seq_length]
    stats["num_active_layers"] += [num_active_layers[-1]]

    # Publish
    cramming.utils.wandb_log(stats, cfg)
    log.info(log_msg)

    # Clear:
    loss_vals = []
    train_time = time.time()
    return loss_vals, train_time


def flag_communication(training_allowed):
    """A quick and dirty communication through NCCL. Should not be a major burden."""
    if torch.distributed.is_initialized():
        comm_tensor = torch.as_tensor(training_allowed).cuda()
        torch.distributed.all_reduce(comm_tensor, torch.distributed.ReduceOp.MIN, async_op=False)
        if comm_tensor >= 1:
            return True
        else:
            return False
    else:
        return training_allowed


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_training_process, job_name="pretraining")


if __name__ == "__main__":
    launch()
