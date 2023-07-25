from typing import Literal, Optional

Task = Literal["bert"]

# BERT 16 layer on NVIDIA 3090
# FORWARD AND BACKWARD; here, we track the time of a minibatch size
NUM_LAYERS_AND_BATCH_TO_TIME_FB_BERT = {
    1536: [
        0.02521,
        0.03837,
        0.05177,
        0.06508,
        0.07847,
        0.09188,
        0.01052,
        0.1186,
        0.1319,
        0.1454,
        0.1589,
        0.1722,
        0.1857,
        0.1988,
        0.212,
        0.2253,
    ],
}

# BERT 16 layer on NVIDIA 3090
# FORWARD ONLY (relevant for RhoLoss); here, we track the time of a microbatch size
NUM_LAYERS_AND_BATCH_TO_TIME_F_BERT = {128: 0.09285}


def get_time_per_step(
    batch_size: int, num_active_layers: int, task: Task = "bert", forward_only: bool = False, microbatch_size: Optional[int] = None
) -> float:
    if task == "bert":
        if forward_only:
            time = NUM_LAYERS_AND_BATCH_TO_TIME_F_BERT[batch_size]
        else:
            time = NUM_LAYERS_AND_BATCH_TO_TIME_FB_BERT[batch_size][num_active_layers - 1]

        if microbatch_size is None:
            return time
        else:
            if microbatch_size % 128 != 0 or microbatch_size <= 0:
                raise ValueError("Microbatch size must be multiple of 128")
            return time * microbatch_size / 128.0
    else:
        raise NotImplementedError("Only BERT is supported in this module.")
