import time

import evaluate
import torch
from datasets.iterable_dataset import IterableDataset

from ..models.progressive import get_stacking_scheduler
from .copied import AdamWScale
from .lion import Lion
from .logging import Averager
from .sophia import SophiaG


def maybe_save_checkpoint(accelerator, args):
    if args.debug:
        return

    if (
        args.seconds_counter > args.budget * 3600
        or args.seconds_counter >= args.every_seconds * args.check_cou
    ):
        args.check_cou += 1
        output_dir = f"checkpoint-{args.mode}-{args.seconds_counter}"
        accelerator.save_state(output_dir=output_dir)


def maybe_eval_predict(model, dataloader, logger, args, tokenizer):
    if (
        args.seconds_counter > args.budget * 3600
        or args.seconds_counter >= args.every_seconds * args.eval_cou
    ):
        args.eval_cou += 1
        model.eval()

        if args.stacking.typ == "drop":
            model.set_active_layers(args.stacking.num_layers_to_add)

        with torch.no_grad():
            eval(model, dataloader, logger, args, tokenizer)

            if args.mode == "ft":
                predict(model, dataloader, logger, args, tokenizer)

        args.last_log = time.time()
        model.train()


def maybe_logging(averager, args, model, optimizer, logger):
    if args.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, model, optimizer)

        seconds_per_step = (time.time() - args.last_log) / args.logging.every_steps
        stats["seconds_per_step"] = seconds_per_step

        stats["real_step"] = args.current_train_step
        stats["seconds_counter"] = args.seconds_counter
        stats["hours_counter"] = args.seconds_counter / 3600
        stats["hours_since_start"] = (time.time() - args.start_time) / 3600

        averager.update(stats)
        averaged_stats = averager.average()

        logger.log_stats(
            stats=averaged_stats,
            step=int(args.fake_step),
            args=args,
            prefix=args.logging.prefix + "train/",
        )

        args.last_log = time.time()


def maybe_grad_clip_and_grad_calc(accelerator, model, args):
    grad_l2 = None

    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        )

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                sum(
                    p.grad.detach().data.norm(2).item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )

        return {"grad_l2": grad_l2}
    else:
        return {}


def extra_stats(args, model, optimizer):
    stats = {}

    if args.logging.weights_l2:
        weights_l2 = (
            sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        )
        stats["weights_l2"] = weights_l2

    cur_lr = optimizer.param_groups[0]["lr"]
    stats["lr"] = cur_lr

    return stats


def forward(model, batch, calc_acc=False):
    stats = {}

    outputs = model(**batch)

    stats["loss"] = outputs.loss.detach().float().item()
    combined_loss = outputs.loss

    if calc_acc:
        correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
        accuracy = correct / batch["labels"].numel()
        stats["accuracy"] = accuracy

    return combined_loss, stats


def eval(model, dataloader, logger, args, tokenizer):
    args.last_log = time.time()
    averager = Averager()

    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.eval.corrected_steps * args.optim.grad_acc:
            break

        _, stats = forward(model, batch, calc_acc=True)
        averager.update(stats)

    averager.update({"time": time.time() - args.last_log})
    averaged_stats = averager.average()
    averaged_stats["num_active_layers"] = model.get_num_active_layers()

    logger.log_stats(
        stats=averaged_stats,
        args=args,
        step=int(args.fake_step),
        prefix=args.logging.prefix + "eval/",
    )


def predict(model, dataloader, logger, args, tokenizer):
    args.last_log = time.time()
    metric = evaluate.load("rouge")
    samples_seen = 0

    def decode(preds):
        preds[preds == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds = [pred.strip() for pred in preds]
        return preds

    for step, batch in enumerate(dataloader):
        predictions = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=args.data.max_target_len,
            generation_config=model.generation_config,
        )
        predictions = decode(predictions)
        references = decode(batch["labels"])

        # If we are in a multiprocess environment, the last batch has duplicates
        if step == len(dataloader) - 1:
            predictions = predictions[: len(dataloader.dataset) - samples_seen]
            references = references[: len(dataloader.dataset) - samples_seen]
        else:
            samples_seen += len(references)

        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute(use_stemmer=True, use_aggregator=False)
    rougeL = sum(eval_metric["rougeL"]) * 100 / len(eval_metric["rougeL"])

    logger.log_stats(
        stats={
            "rougeL": rougeL,
            "time": time.time() - args.last_log,
            "num_active_layers": model.get_num_active_layers(),
        },
        args=args,
        step=int(args.fake_step),
        prefix=args.logging.prefix + "test/",
    )


def _get_fake_step_seconds(seconds_elapsed, seconds_budget, num_training_steps):
    if seconds_elapsed == 0:
        fake_step = 0
    else:
        fake_step = int(seconds_elapsed / seconds_budget * num_training_steps)
    return fake_step


NUM_LAYERS_AND_BATCH_TO_TIME_T5 = {
    144: {
        1: 0.129,
        2: 0.209,
        3: 0.291,
        4: 0.373,
        5: 0.455,
        6: 0.538,
        7: 0.619,
        8: 0.703,
        9: 0.783,
        10: 0.865,
        11: 0.947,
        12: 1.031,
    },
}


def get_time_per_step(batch_size: int, num_active_layers: int) -> float:
    return NUM_LAYERS_AND_BATCH_TO_TIME_T5[batch_size][num_active_layers]


def get_optimizer_time(optimizer, args):
    assert args.stacking.typ == "none"

    if isinstance(optimizer.optimizer, SophiaG):
        return 1.156
    elif isinstance(optimizer.optimizer, Lion):
        return 1.043
    elif isinstance(optimizer.optimizer, AdamWScale):
        return 1.031
    else:
        raise NotImplementedError


def train(
    model,
    train_dataloader,
    test_dataloader,
    accelerator,
    lr_scheduler,
    optimizer,
    logger,
    args,
    tokenizer,
):
    model.train()

    train_averager = Averager()

    stacking_scheduler = get_stacking_scheduler(
        model=model,
        optimizer=optimizer,
        num_active_layers=args.stacking.num_initial_layers,
        num_layers_to_add=args.stacking.num_layers_to_add,
        num_train_steps=args.optim.total_steps,
        typ=args.stacking.typ,
        step_fractions=args.stacking.step_fractions,
        doubling=args.stacking.doubling,
        seconds_budget=args.budget * 3600,
        gamma_factor=args.stacking.gamma_factor,
    )

    sophia_update, sophia_batches = False, 0

    while args.seconds_counter <= args.budget * 3600:
        if isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(args.current_epoch)

        optimizer.zero_grad(set_to_none=True)

        for batch_id, batch in enumerate(train_dataloader, start=1):
            if args.seconds_counter > args.budget * 3600:
                break

            if sophia_update:
                outputs = model(**batch)
                samp_dist = torch.distributions.Categorical(logits=outputs.logits)
                y_sample = samp_dist.sample()
                loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                    outputs.logits.view(-1, outputs.logits.size(-1)), y_sample.view(-1)
                )
                accelerator.backward(loss / args.optim.grad_acc)
                sophia_batches += 1

                if sophia_batches == args.optim.grad_acc:
                    maybe_grad_clip_and_grad_calc(accelerator, model, args)
                    optimizer.optimizer.update_hessian()
                    optimizer.zero_grad(set_to_none=True)
                    sophia_update, sophia_batches = False, 0

                continue

            loss, stats = forward(model, batch)
            accelerator.backward(loss / args.optim.grad_acc)
            train_averager.update(stats)

            if batch_id % args.optim.grad_acc == 0:
                stats = maybe_grad_clip_and_grad_calc(accelerator, model, args)
                train_averager.update(stats)
                train_averager.update(
                    {"num_active_layers": model.get_num_active_layers()}
                )

                optimizer.step()
                lr_scheduler.step(args.fake_step)
                stacking_scheduler.step(args.seconds_counter)
                optimizer.zero_grad(set_to_none=True)

                maybe_logging(train_averager, args, model, optimizer, logger)
                maybe_save_checkpoint(accelerator, args)
                maybe_eval_predict(model, test_dataloader, logger, args, tokenizer)

                if (
                    isinstance(optimizer.optimizer, SophiaG)
                    and args.current_train_step % args.sophia_freq == 0
                ):
                    sophia_update = True

                args.current_train_step += 1
                if args.stacking.typ == "none":
                    args.seconds_counter += get_optimizer_time(optimizer, args)
                else:
                    args.seconds_counter += get_time_per_step(
                        args.optim.batch_size, model.get_num_active_layers()
                    )
                args.fake_step = _get_fake_step_seconds(
                    args.seconds_counter, args.budget * 3600, args.optim.total_steps
                )

        args.current_epoch += 1

    maybe_save_checkpoint(accelerator, args)
    maybe_eval_predict(model, test_dataloader, logger, args, tokenizer)
