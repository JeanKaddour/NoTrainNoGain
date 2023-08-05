"""Script for a pretraining run with selective backpropagation."""

import datetime
import logging
import os
import time
from collections import defaultdict, deque

import hydra
import torch
from scipy.stats import percentileofscore

import cramming
import cramming.utils
from cramming.utils import validate
from rst.saved_rsts import get_time_per_step

log = logging.getLogger(__name__)


class SBForward:
    """
    Short implementation of selective backpropagation.
    This function is supposed to replace default model.forward()

    Whenever in your code you're doing model.forward(), you should do
    FancyForward_object.forward() instead.

    FancyForward will accumulate examples and labels in a batch_acc with (no_grad) selective forwards
    When it has accumulated enough examples, it will do a proper forward

    Selective forwards returns a tuple (None, None), proper forwards returns loss and stats
    """

    def __init__(self, mini_batch_size: int, micro_batch_size: int, scale: float):
        self.mini_batch_size = mini_batch_size
        self.micro_batch_size = micro_batch_size
        self.scale = scale  # selective scale (1 = 50%, 2 = 33%)
        self.batch_acc = {
            "input_ids": torch.tensor([], device="cpu").long(),
            "labels": torch.tensor([], device="cpu").long(),
        }
        self.historical_loss = deque(maxlen=5000)

    def split_minibatch_into_microbatches(self, minibatch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Create a list of micro batch dicts from a minibatch dict."""
        micro_batches = list()
        for i in range(0, len(minibatch["input_ids"]), self.micro_batch_size):
            micro_batches.append({k: minibatch[k][i : i + self.micro_batch_size] for k in minibatch.keys()})
        return micro_batches

    def proper_forward(self):
        if self.batch_acc["input_ids"].size(0) < self.mini_batch_size:
            return None
        else:
            mini_batch = {k: v[: self.mini_batch_size] for k, v in self.batch_acc.items()}
            self.batch_acc = {k: v[self.mini_batch_size :] for k, v in self.batch_acc.items()}
            return self.split_minibatch_into_microbatches(mini_batch)

    def acc(self, batch):
        for k, v in batch.items():
            self.batch_acc[k] = torch.cat(
                [self.batch_acc[k], v.cpu()],
                dim=0,
            )

    def forward(self, batch, losses):
        self.historical_loss.extend(losses.detach().cpu().numpy().tolist())
        percentiles = percentileofscore(self.historical_loss, losses.detach().cpu().numpy().tolist()) / 100.0
        # We sample from Bernoulli if we take the example or not
        # We use the scale to make it more or less selective (details in the paper)

        percentiles = torch.tensor(percentiles, device="cpu")
        sample = torch.distributions.bernoulli.Bernoulli(probs=percentiles**self.scale).sample().bool()

        # Subsample the batch based on the Bernoulli outputs
        batch = {k: v[sample] for k, v in batch.items()}

        # Accummulate
        self.acc(batch)

        # Proper forward if we've reached the batch size, otherwise return None
        return self.proper_forward()


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
    training_allowed = True
    loss, loss_vals = None, []
    num_active_layers: list[int] = []
    seconds_budget = 60 * 60 * cfg.train.budget  # 1 day by default

    iterable_data = enumerate(dataloader)

    forward = SBForward(
        mini_batch_size=cfg.train.batch_size,
        micro_batch_size=cfg.impl.microbatch_size,
        scale=cfg.train.sb.scale,
    )

    # Launch training
    for step, batch in iterable_data:
        step_seconds_counter = seconds_counter if "seconds" in cfg.train.scheduler else None

        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)
        losses = model_engine.forward_losses(device_batch)
        if cfg.train.track_forward_pass_only:
            seconds_counter += get_time_per_step(cfg.impl.microbatch_size, model_engine.get_num_active_layers(), forward_only=True)
        num_active_layers.append(model_engine.get_num_active_layers())

        if seconds_counter >= seconds_budget:
            break
        micro_batches = forward.forward(device_batch, losses)
        if micro_batches is not None:
            for batch in micro_batches:
                device_batch = model_engine.to_device(batch)
                loss = model_engine.step(device_batch, step_seconds_counter)
                seconds_counter += get_time_per_step(cfg.impl.microbatch_size, model_engine.get_num_active_layers(), forward_only=False)
                loss_vals.append(loss.detach())
                if seconds_counter >= seconds_budget:
                    break

        if cfg.arch.layer_drop.enabled:
            # If we evaluate we want to use all the layers.
            model.encoder.active_layer_indices = []

        # Check stopping criteria
        # if check_deadline(wallclock_timer, cfg.budget) or step == cfg.train.steps:
        if not within_budget(cfg, seconds_counter, step):
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


def within_budget(cfg, seconds_counter: float, step_counter: int):
    if cfg.budget == "steps":
        return step_counter < cfg.train.steps
    else:
        # In this case the budget is in hours.
        seconds_budget = 60 * 60 * cfg.train.budget
        return seconds_counter < seconds_budget


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
