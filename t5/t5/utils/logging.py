import logging
import os
from collections import defaultdict

import datasets
import neptune
import transformers
import wandb
from accelerate.logging import get_logger
from neptune.utils import stringify_unsupported
from omegaconf import OmegaConf, open_dict


class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats


class Logger:
    def __init__(self, args, accelerator):
        self.logger = get_logger("Main")

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f"Working directory is {os.getcwd()}")

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        self.setup_neptune(args)
        self.setup_wandb(args)

    def setup_wandb(self, args):
        if args.logging.wandb:
            wandb.init(
                name=args.logging.wandb_creds.name,
                project=args.logging.wandb_creds.project,
                entity=args.logging.wandb_creds.entity,
            )
        else:
            self.wandb_run = None

        self.wandb_run = wandb.run

        with open_dict(args):
            if self.wandb_run is not None:
                args.wandb_id = self.wandb_run.id

    def setup_neptune(self, args):
        if args.logging.neptune:
            tags = [str(item) for item in args.logging.neptune_creds.tags.split(",")]
            if tags == [] or tags == [""]:
                tags = None

            neptune_logger = neptune.init_run(
                project=args.logging.neptune_creds.project,
                api_token=args.logging.neptune_creds.api_token,
                tags=tags,
            )
        else:
            neptune_logger = None

        self.neptune_logger = neptune_logger

        with open_dict(args):
            if neptune_logger is not None:
                args.neptune_id = neptune_logger["sys/id"].fetch()

    def log_args(self, args):
        if self.wandb_run is not None:
            logging_args = OmegaConf.to_container(args, resolve=True)
            wandb.config.update(logging_args)

        if self.neptune_logger is not None:
            logging_args = OmegaConf.to_container(args, resolve=True)
            self.neptune_logger["args"] = stringify_unsupported(logging_args)

    def log_stats(self, stats, step, args, prefix=""):
        if self.neptune_logger is not None:
            for k, v in stats.items():
                self.neptune_logger[f"{prefix}{k}"].log(v, step=step)

        if self.wandb_run is not None:
            for k, v in stats.items():
                wandb.log({f"{prefix}{k}": v}, step=step)

        msg_start = (
            f"[{prefix[:-1]}] Step {step} out of {args.optim.total_steps}" + " | "
        )
        dict_msg = (
            " | ".join([f"{k.capitalize()} --> {v:.3f}" for k, v in stats.items()])
            + " | "
        )

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)

    def finish(self):
        if self.neptune_logger is not None:
            self.neptune_logger.stop()

        if self.wandb_run is not None:
            wandb.finish()
