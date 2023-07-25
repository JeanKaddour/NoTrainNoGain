"""Evaluates a pretrained model on the pretraining validation set.

Optionally updates the wandb run with the validation loss.
"""

import logging
import sys

import hydra
import torch
import wandb
from wandb.apis.public import Run

import cramming

log = logging.getLogger(__name__)
from cramming.utils import validate


def main_eval_process(cfg, setup):
    """This function controls the central training loop."""

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    model = cramming.construct_model(cfg_arch, tokenizer.vocab_size)
    train_dataset, validation_set, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl, cfg.train)
    if cfg.truncate_dataset > 0:
        train_dataset = train_dataset.select(range(min(cfg.truncate_dataset, len(train_dataset))))
        validation_set = validation_set.select(range(min(cfg.truncate_dataset, len(validation_set))))
    log.info(f"Train dataset size: {len(train_dataset)}, validation set size: {len(validation_set)}")
    model_engine, _, _, _, validation_loader = cramming.load_backend(
        model,
        train_dataset,
        validation_set,
        tokenizer,
        cfg.train,
        cfg.impl,
        setup=setup,
    )

    model_engine.load_checkpoint(cfg_arch, model_file)
    model_engine.eval()
    validation_loss = [validate(model_engine, validation_loader, model_engine.setup["device"])]

    log.info(f"Avg Validation loss: {validation_loss}")

    if cfg.wandb.resume is not None:
        print(f"Would you like to update existing run {cfg.wandb.resume}?")
        if _ask_yes_no():
            logged_run: Run = wandb.Api().run(path=f"{cfg.wandb.entity}/{cfg.wandb.project}/{cfg.wandb.resume}")
            hour_to_log = logged_run.history(keys=["hours"])["hours"].values[-1] + 0.0001
            print(f"Logging at hour {hour_to_log:.3f}")
            wandb.log({"validation_loss": validation_loss, "hours": hour_to_log})
        else:
            print("Not logging")


def _ask_yes_no() -> bool:
    while True:
        sys.stdout.write("y/n:")
        response = input().lower()
        if response == "y":
            return True
        if response == "n":
            return False


@hydra.main(config_path="cramming/config", config_name="cfg_eval_pt", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_eval_process, job_name="eval_pt_task")


if __name__ == "__main__":
    launch()
