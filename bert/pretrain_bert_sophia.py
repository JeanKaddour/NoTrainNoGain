"""Script for a pretraining run using Sophia."""

import datetime
import logging
import os
import time
from collections import defaultdict

import hydra
import torch

import cramming
import cramming.utils
from cramming.utils import validate
from efficient_training import layer_drop
from rst.saved_rsts import get_time_per_step

log = logging.getLogger(__name__)


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
    loss_vals = []
    num_active_layers: list[int] = []
    sophia_updates_enabled = False
    sophia_acc_batches = 0
    sophia_cross_entropy = torch.nn.CrossEntropyLoss()
    sophia_latest_update_step = -1
    loss = None
    iterable_data = enumerate(dataloader)
    if cfg.train.gradinit.enabled:
        model_engine.gradinit(iterable_data, cfg.train.optim, cfg.train.gradinit)

    # Launch training
    for step, batch in iterable_data:
        if cfg.arch.layer_drop.enabled:
            assert not cfg.train.stacking.enabled, "Cannot layer drop and stack at same time"
            # Must do this before incrementing step counter, so model_engine.get_num_active_layers() is correct.
            model.encoder.active_layer_indices, layer_drop_prob = layer_drop.sample_active_layers(seconds_counter, step, cfg)
        else:
            layer_drop_prob = None

        # We save this here for logging, because during layer drop we enable all the layers again below.
        num_active_layers.append(model_engine.get_num_active_layers())

        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)

        step_seconds_counter = seconds_counter if "seconds" in cfg.train.scheduler else None

        if (
            sophia_latest_update_step != model_engine.update_steps
            and model_engine.update_steps % cfg.train.sophia.hess_update_frequency == 0
        ):
            sophia_updates_enabled = True
        if sophia_updates_enabled:
            model_engine.optimizer.zero_grad(set_to_none=True)
            # update hessian EMA
            logits = model_engine.forward(**device_batch)["logits"]
            samp_dist = torch.distributions.Categorical(logits=logits)
            y_sample = samp_dist.sample()
            loss_sampled = sophia_cross_entropy(logits, y_sample)
            loss_sampled.backward()
            model_engine.scheduler.step(step_seconds_counter)
            sophia_acc_batches += 1
            if sophia_acc_batches * cfg.impl.microbatch_size >= cfg.train.sophia.batch_size_hess_update:
                sophia_updates_enabled = False
                sophia_acc_batches = 0
                model_engine.optimizer.update_hessian()
                model_engine.optimizer.zero_grad(set_to_none=True)
                sophia_latest_update_step = model_engine.update_steps

            if not cfg.train.sophia.free_updates:
                seconds_counter += get_time_per_step(
                    model_engine.current_batch_size, model_engine.get_num_active_layers(), microbatch_size=cfg.impl.microbatch_size
                )

        else:
            loss = model_engine.step(device_batch, step_seconds_counter)
            loss_vals.append(loss.detach())

            seconds_counter += get_time_per_step(
                model_engine.current_batch_size, model_engine.get_num_active_layers(), microbatch_size=cfg.impl.microbatch_size
            )

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
        elif step > 0 and step % cfg.impl.print_loss_every_nth_step == 0:
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
            if check_early_termination(wallclock_timer, stats["loss"][-1], cfg.impl.early_termination):
                training_allowed = False
                log.info("Loss higher than allowed threshold. Stopping training early...")

        if layer_drop_prob is not None:
            stats["layer_drop_prob"] += [layer_drop_prob]

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
    weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model_engine.model.parameters()) ** 0.5
    stats["weights_l2"] += [weights_l2]
    stats["step"] += [step]
    stats["seconds"] += [seconds_counter]
    stats["hours"] += [seconds_counter / 3600]

    stats["epoch"] += [dataloader.epoch_counter]

    tokens_per_step = cramming.utils.num_processes() * model_engine.record_tokens_per_step()
    stats["tokens"] += [step * tokens_per_step]
    stats["loss"] += [torch.stack(loss_vals).mean().item()]  # Averaged loss
    stats["update_steps"] += [model_engine.update_steps]
    current_lr = model_engine.optimizer.param_groups[0]["lr"]
    log_msg = f"Train loss {loss_vals[-1].item():2.4f} at step {step} with lr {current_lr:.5f}. "
    log_msg += f"[Avg: {stats['loss'][-1]:2.4f}] "
    if step > 0:
        stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
        estimated_train_finish = str(datetime.timedelta(seconds=stats["train_time"][-1] * cfg.train.steps))
        tokens_per_second = tokens_per_step / stats["train_time"][-1]
        stats["tok/sec"] += [int(tokens_per_second)]
        log_msg += f" Perf: {stats['train_time'][-1]:2.4f}s per step ({tokens_per_second:.0f}t/s). "
        log_msg += f"Estimated Total Train: {estimated_train_finish}."

        num_param = 0
        num_effective = 0
        hessian_norm2 = 0

        LL = len(model_engine.optimizer.state_dict()["state"])

        for jj in range(LL):
            num_param += model_engine.optimizer.state_dict()["state"][jj]["exp_avg"].numel()
            num_effective += torch.sum(
                torch.abs(model_engine.optimizer.state_dict()["state"][jj]["exp_avg"])
                < cfg.train.optim.rho * cfg.train.optim.bs * model_engine.optimizer.state_dict()["state"][jj]["hessian"]
            )
            hessian_norm2 += model_engine.optimizer.state_dict()["state"][jj]["hessian"].detach().norm(2).item() ** 2
        stats["hessian_norm2"] += [hessian_norm2**0.5]
        stats["num_param"] += [num_param]
        stats["num_effective"] += [num_effective]

    # Adaptive optim stats
    stats["lr"] += [current_lr]
    stats["batch_size"] += [model_engine.record_batch_size()]
    stats["seq_length"] = [model_engine.current_seq_length]
    stats["num_active_layers"] += [num_active_layers[-1]]
    stats["avg_num_active_layers"] += [sum(num_active_layers) / len(num_active_layers)]

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
