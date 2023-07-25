def get_optimizer(model, args):
    if args.optim.name == "adamwscale":
        from .copied import AdamWScale

        optimizer = AdamWScale(
            model.parameters(),
            lr=args.optim.base_lr,
        )
    elif args.optim.name == "lion":
        from .lion import Lion

        optimizer = Lion(
            model.parameters(),
            weight_decay=args.optim.weight_decay,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == "sophia":
        from .sophia import SophiaG

        optimizer = SophiaG(
            model.parameters(),
            rho=args.optim.rho,
            weight_decay=args.optim.weight_decay,
            lr=args.optim.base_lr,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args, logger):
    if args.optim.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps],
        )
    elif args.optim.lr_scheduler == "legacy":
        import math

        from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(1e-2, 1.0 / math.sqrt(step)) / args.optim.base_lr
            if step
            else 1e-2 / args.optim.base_lr,
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1],
        )
    elif args.optim.lr_scheduler == "constant":
        from transformers import get_scheduler

        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    elif args.optim.lr_scheduler == "cosine-budget":
        import math

        from torch.optim.lr_scheduler import LambdaLR

        num_warmup_steps = args.optim.warmup_steps
        num_training_steps = args.optim.total_steps
        num_cycles = 0.5

        def lr_lambda(current_step):
            fake_step = current_step

            if fake_step < num_warmup_steps:
                return (
                    (float(fake_step) / float(max(1, num_warmup_steps))) * 0.5
                ) + 0.5

            progress = float(fake_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                1e-5,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )

        return LambdaLR(optimizer, lr_lambda, -1)
    else:
        raise NotImplementedError

    return lr_scheduler
