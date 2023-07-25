import time

import hydra
from accelerate import Accelerator
from omegaconf import open_dict
from torch import compile, no_grad

from .models import get_config, get_model, get_tokenizer
from .utils.data import get_dataloaders
from .utils.general import setup_basics
from .utils.optim import get_lr_scheduler, get_optimizer
from .utils.train import eval, predict, train


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(args):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
    )
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    model = get_model(args, config)
    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    if args.model.compile:
        if args.stacking.typ == "none":
            model = compile(model)
        else:
            model.lm_head = compile(model.lm_head)
            model.shared = compile(model.shared)
            model.encoder.embed_tokens = compile(model.encoder.embed_tokens)
            model.decoder.embed_tokens = compile(model.decoder.embed_tokens)
            model.decoder.final_layer_norm = compile(model.decoder.final_layer_norm)
            for i in range(len(model.encoder.block)):
                model.encoder.block[i] = compile(model.encoder.block[i])

            for i in range(len(model.decoder.block)):
                model.decoder.block[i] = compile(model.decoder.block[i])

    with open_dict(args):
        args.start_time = time.time()
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()
        args.seconds_counter = 0.0
        args.fake_step = 0
        args.eval_cou = 1
        args.check_cou = 1

    if args.eval_only:
        model.eval()
        with no_grad():
            eval(model, test_dataloader, logger, args, tokenizer)
    elif args.predict_only:
        model.eval()
        with no_grad():
            predict(model, test_dataloader, logger, args, tokenizer)
    else:
        train(
            model,
            train_dataloader,
            test_dataloader,
            accelerator,
            lr_scheduler,
            optimizer,
            logger,
            args,
            tokenizer,
        )

    logger.finish()


if __name__ == "__main__":
    main()
