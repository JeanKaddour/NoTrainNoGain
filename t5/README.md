# T5 experiments
The T5 experiments are based off the excellent [nanoT5](https://github.com/PiotrNawrot/nanoT5) repository, see [LICENSE](LICENSE).

## Environment setup

Following nanoT5's setup:

```
conda create -n ntng_t5 python=3.8
conda activate ntng_t5
pip install -r requirements.txt
```

The following commands result in the following [pip freeze](assets/pip_freeze.txt) as of 24.07.2023. We also include our [lscpu](assets/lscpu.txt) and [nvidia-smi](assets/nvidia_smi.txt).

## Commands for each experiment

By default the experiments are run for 24 hours. For more details check the default config with all hyperparameters [here](t5/configs/default.json). We include the RST measurements [here](t5/utils/train.py).

### Baseline

```
    python -m t5.train stacking.typ=none
```

### Stacking

```
    python -m t5.train stacking.typ=stack
```

### Layer Dropping

```
    python -m t5.train stacking.typ=drop optim.base_lr=1e-2 stacking.gamma_factor=20
```

### Sophia

```
    python -m t5.train stacking.typ=none optim.name=sophia optim.rho=1e-2 optim.base_lr=1e-3 sophia_freq=10
```

### Lion

```
    python -m t5.train stacking.typ=none optim.name=lion optim.base_lr=7.5e-4
```

### Fine-Tuning

We fine-tune the models in the original [nanoT5 repository](https://github.com/PiotrNawrot/nanoT5) using the following command:

```

    python -m nanoT5.main task=ft google/t5-v1_1-base model.random_init=false model.checkpoint_path="/path/to/pytorch_model.bin
```

All our models do not modify the original T5 architecture, so all checkpoints trained in this repository are compabible with the original nanoT5 repository.