"""Script to fine tune and evaluate a pretrained model."""
import copy
import datetime
import logging
import time
from collections import defaultdict

import evaluate
import hydra
import torch
import wandb

import cramming

log = logging.getLogger(__name__)


def main_downstream_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    tasks = cramming.prepare_task_dataloaders(tokenizer, cfg.eval, cfg.impl)

    metrics = dict()
    stats = defaultdict(list)
    # Start the clocks now:
    for task_name, task in tasks.items():
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        # Prepare model for finetuning:
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        model_engine, _, _, _, _ = cramming.load_backend(model, None, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        model_engine.load_checkpoint(cfg_arch, model_file)
        metric = evaluate.load(task["details"]["collection"], task_name, cache_dir=cfg.impl.path)
        # Launch training
        model_engine.train()
        loss_vals = []
        for epoch in range(cfg.eval.epochs):
            train_time = time.time()
            for step, batch in enumerate(task["trainloader"]):
                # Heavy lifting is moved to engines
                device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
                loss = model_engine.step(device_batch)
                loss_vals.append(loss.detach())
                if cfg.dryrun:
                    break

            metrics[task_name] = validate(model_engine, task["validloader"], metric, setup, cfg, task_name)
            stats[f"{task_name}_epoch"] += [epoch]
            stats[f"{task_name}_loss"] += [loss.item()]

            stats[f"{task_name}_avg_loss"] += [torch.stack(loss_vals).mean().item()]  # Smoothed loss
            loss_vals = []
            current_lr = model_engine.optimizer.param_groups[0]["lr"]

            log_msg = f"Train loss {loss.item():2.4f} at step {step} with lr {current_lr:.5f}. "
            log_msg += f"[Avg: {stats[f'{task_name}_avg_loss'][-1]:2.4f}] after epoch {epoch}."

            stats[f"{task_name}_train_time"] += [(time.time() - train_time)]
            estimated_train_finish = str(datetime.timedelta(seconds=stats[f"{task_name}_train_time"][-1] * cfg.eval.epochs))
            tokens_per_second = (step + 1) * cfg.eval.max_seq_length * cfg.impl.microbatch_size / stats[f"{task_name}_train_time"][-1]
            log_msg += (
                f" Perf: {stats[f'{task_name}_train_time'][-1]/60:2.4f}min per epoch ({tokens_per_second:.0f}t/s). "
                f"Estimated Total Train: {estimated_train_finish}."
            )

            for name, metric_val in metrics[task_name].items():
                stats[f"{task_name}_{name}"] += [metric_val]
            log.info(log_msg)
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in metrics[task_name].items()])
            log.info(f"Validation metric is {msg_metrics} after epoch {epoch}.")
            cramming.utils.wandb_log(stats, cfg)

            if cfg.dryrun:
                break
        # Launch testing:
        if task["extra_validloader"] is not None:
            extra_eval_metric = validate(model_engine, task["extra_validloader"], metric, setup, cfg, task_name)
            metrics[task_name + "extra"] = extra_eval_metric
            for name, metric_val in extra_eval_metric.items():
                stats[f"{task_name}_{name}_extra"] += [metric_val]
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in extra_eval_metric.items()])
            log.info(f"Extra validation metric is {msg_metrics} after finetuning.")
            cramming.utils.wandb_log({f"{task_name}_{name}_extra": [v] for k, v in extra_eval_metric.items()}, cfg)

    final_metrics = {}
    avg_score = 0.0
    metrics_to_average = copy.deepcopy(cfg.eval.metrics_to_average)
    for name, metric in stats.items():
        if name in metrics_to_average:
            final_metrics[name] = max(metric)
            avg_score += final_metrics[name]
            metrics_to_average.remove(name)
    if len(metrics_to_average) > 0:
        log.warning(f"Metrics {metrics_to_average} were not found in {stats.keys()}.")
    final_metrics["avg_glue_scores"] = avg_score / len(final_metrics)
    log.info(f"Final metrics are {final_metrics}.")
    wandb.log({"final": final_metrics})
    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.dump_metrics(cfg, metrics)
        cramming.utils.save_summary("downstream", cfg, metrics, stats, time.time() - local_time, setup)


@torch.no_grad()
def validate(model_engine, validloader, metric, setup, cfg, task_name=""):
    """Evaluate on validation set."""
    model_engine.eval()
    for step, batch in enumerate(validloader):
        device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask", "p_idx", "q_idx", "a_idx"])
        forward_batch = {k: v for k, v in device_batch.items() if k not in ["p_idx", "q_idx", "a_idx"]}

        _, predictions = model_engine.forward_inference(**forward_batch)
        if task_name == "multirc":
            predictions = [
                {"prediction": p, "idx": {"paragraph": p_idx, "question": q_idx, "answer": a_idx}}
                for p, p_idx, q_idx, a_idx in zip(predictions, device_batch["p_idx"], device_batch["q_idx"], device_batch["a_idx"])
            ]
        else:
            # Make sure predictions has the same dtype as the reference, particularly if we
            # are using bf16.
            predictions = predictions.to(device_batch["labels"])

        metric.add_batch(predictions=predictions, references=device_batch["labels"])
        if cfg.dryrun and step > 1:
            break
    eval_metric = metric.compute()
    model_engine.train()
    return eval_metric


@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")


if __name__ == "__main__":
    launch()
