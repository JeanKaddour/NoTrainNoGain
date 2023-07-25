"""Extracts the RHO-Loss irreducible losses from a model."""
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Optional

import hydra
import numpy as np
import torch
import wandb
from tqdm import tqdm

import cramming

log = logging.getLogger(__name__)


def save_chunk(chunk_data: dict, chunk_number: int, path: str) -> None:
    with open(os.path.join(path, f"dict_chunk_{chunk_number}.pkl"), "wb") as file:
        pickle.dump(chunk_data, file)


def get_example_ids_from_batch(examples_counter, len_batch: int, len_dataset: Optional[int] = None) -> list[int]:
    example_ids = examples_counter + np.arange(len_batch)
    example_ids = example_ids.tolist()
    if len_dataset is not None:
        example_ids = [example_id % len_dataset for example_id in example_ids]
    return example_ids


def save_losses_of_il_model(cfg, setup):
    """This function controls the central training loop."""
    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    model = cramming.construct_model(cfg.arch, tokenizer.vocab_size)
    train_set, validation_set, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl, cfg.train)
    model_engine, _, _, train_dataloader, validation_dataloader = cramming.load_backend(
        model,
        train_set,
        validation_set,
        tokenizer,
        cfg.train,
        cfg.impl,
        setup=setup,
    )
    model_engine.load_checkpoint(cfg_arch, model_file)
    model_engine.eval()
    iterable_data = enumerate(tqdm(train_dataloader))
    path = os.path.join(cfg.base_dir, cfg.name, "examples_to_loss")
    os.makedirs(path, exist_ok=True)
    log.info(f"Saving losses of IL model to {path}")
    train_time = time.time()  # Crude time measurement for print_loss_every_nth_step
    stats = defaultdict(list)

    # Launch training
    examples_to_loss_dict = {}
    chunk_counter = 0
    example_ids = []
    examples_counter = 0
    with torch.no_grad():
        for step, batch in iterable_data:
            # Heavy lifting is moved to engines
            example_ids_in_batch = get_example_ids_from_batch(examples_counter, len(batch["input_ids"]))
            example_ids.extend(example_ids_in_batch)
            device_batch = model_engine.to_device(batch)
            examples_counter += len(batch["input_ids"])
            with torch.autocast(**model_engine.amp_settings):
                losses = model_engine.model.forward_all_losses(**device_batch)
            examples_to_loss_dict.update(dict(zip(example_ids_in_batch, losses)))
            if step > 0 and step % cfg.impl.saving_interval == 0:
                examples_to_loss_dict = {k: v.cpu().tolist() for k, v in examples_to_loss_dict.items()}
                save_chunk(examples_to_loss_dict, chunk_counter, path)
                examples_to_loss_dict = {}  # free up RAM
                chunk_counter += 1
            if step > 0 and step % cfg.impl.print_loss_every_nth_step == 0:
                stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
                stats["step"] += [step]
                stats["examples_counter"] += [examples_counter]
                train_time = time.time()
                wandb.log({k: v[-1] for k, v in stats.items()}, step=stats["step"][-1] if "step" in stats else None)
    examples_to_loss_dict = {k: v.cpu().tolist() for k, v in examples_to_loss_dict.items()}
    save_chunk(examples_to_loss_dict, chunk_counter, path)
    stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
    stats["step"] += [step]
    stats["examples_counter"] += [examples_counter]
    wandb.log({k: v[-1] for k, v in stats.items()}, step=stats["step"][-1] if "step" in stats else None)


@hydra.main(config_path="../cramming/config", config_name="cfg_save_losses", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, save_losses_of_il_model, job_name="save_losses")


if __name__ == "__main__":
    launch()
