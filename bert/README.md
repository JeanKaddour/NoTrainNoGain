# BERT experiments
The BERT experiments are based off the excellent [Cramming](https://github.com/JonasGeiping/cramming) repository, see [LICENSE.txt](LICENSE.txt).

## Environment setup
The project has the following dependencies:
- CUDA toolkit + nvcc 11.7 (required to install [FlashAttention](https://github.com/Dao-AILab/flash-attention))
- Python 3.10
- [Poetry](https://python-poetry.org/)

One way to install the dependencies is using Conda and the provided environment file:
- `conda env update -f conda_env.yaml`
- `conda activate ntng_bert`
- `export CUDA_HOME=$CONDA_PREFIX` (this is required so PyTorch finds the correct nvcc version when building FlashAttention)

Create and activate the Poetry environment:
- Install: `poetry install`
- Activate: `poetry shell`
- Manually install FlashAttention: `pip install --no-build-isolation flash-attn==1.0.9`

## Modules
### Entry points
* `pretrain_bert.py`
  * implements the baseline, layer stacking, layer dropping, and Lion
* `pretrain_bert_sb.py`
  * modified copy of `pretrain_bert.py` that includes selective backpropagation
* `pretrain_bert_rho_loss.py`
  * modified copy of `pretrain_bert.py` that includes RHO-Loss
  * requires the irreducible losses to be extracted first (see command below)
* `pretrain_bert_sophia.py`
  * modified copy of `pretrain_bert.py` that includes Sophia-G
* `eval.py`
  * implements fine-tuning and evaluating a pretrained model
* `validate_bert.py`
  * implements validating a pretrained checkpoint on the validation set

### Other
* `efficient_training` contains additional code for some of the efficient training methods
  * we recommend starting at the entry point scripts to understand how to use the code
* `rst` includes helper code for tracking the Reference System Time (RST) metric
* `ntng_assets` includes download scripts for the C4 Random subset and irreducible losses (for RHO-Loss)
  * see [ntng_assets/README.md](../ntng_assets/README.md) for more details

## Experiment commands
### Pre-train
First, download the randomized subset of the C4 dataset from our Dropbox: `python ntng/download_c4_subset.py`

If you would like to use [Weights & Biases](https://wandb.ai/site), configure this in `cramming/config/wandb/default.yaml`.

#### Dynamic architectures
* Baseline (FP16):
`python pretrain_bert.py name={name} budget={budget in hours} seed={seed}`


* Layer stacking:
`python pretrain_bert.py name={name} budget={budget in hours} seed={seed} train.stacking.enabled=True`

* Layer dropping:
`python pretrain_bert.py name={name} budget={budget in hours} seed={seed} arch.layer_drop.enabled=True`

#### Batch selection
By default the dataset is the randomized subset of C4, but you can also set `data=minipile` or `data=bookcorpus-wikitext`.
Minipile and BCWK will be downloaded automatically from Hugging Face at the start of training.

* Selective backprop:
`python pretrain_bert_sb.py name={name} budget={budget in hours} seed={seed} train.validation_set.fraction=0.2 impl.validate_every_hours=3`
  * To reproduce the ablation where the additional forward passes are not counted against the training budget, add `train.track_forward_pass_only=false`.

##### RHO-loss
To acquire the irreducible losses you can either:
  * Download ours: `python ntng_assets/download_il_losses.py`
  * Train your own irreducible loss model and extract the losses:
    * `python pretrain_bert.py name=il_model budget={budget in hours} train.validation_set.il_model=True train.validation_set.fraction=0.2`
    * `python efficient_training/extract_il_losses.py name=il_model`

Pre-train: `python pretrain_bert_rho_loss.py name={name} budget={budget in hours} seed={seed} data={dataset} train.validation_set.fraction=0.2 impl.validate_every_hours=3 train.rho_loss.il_losses_path={path to irreducible losses for dataset} train.rho_loss.mega_batch_size=3072`

To reproduce the ablation where the additional forward passes are not counted against the training budget, add `train.track_forward_pass_only=false`.

#### Efficient optimizers
We found Sophia was unstable when using FP16, thus for this set of experiments we use BF16.

* Baseline (BF16):
`python pretrain_bert.py name={name} budget={budget in hours} seed={seed} impl.mixed_precision_target_dtype=bfloat16`

* Lion: `python pretrain_bert.py name={name} budget={budget in hours} seed={seed}  impl.mixed_precision_target_dtype=bfloat16 train/optim=lion train.optim.lr={learning rate} train.optim.weight_decay={weight decay}`

* Sophia: `python pretrain_bert_sophia.py name={name} budget={budget in hours} seed={seed} impl.mixed_precision_target_dtype=bfloat16 train/optim=sophiag train.optim.rho={Sophia rho} train.optim.lr={learning rate} train.optim.weight_decay={weight decay}`
  * To reproduce the ablation where the additional forward passes are not counted against the training budget, add `train.sophia.free_updates=True`.



### Fine tune & evaluate
Fine tune and evaluate a checkpoint using GLUE:

`python eval.py name={pretrain name} eval=glue_sane impl.microbatch_size=16 impl.shuffle_in_dataloader=true seed=0 [impl.mixed_precision_target_dtype=bfloat16 if the checkpoint was trained using BF16 rather than FP16]`

Fine tune and evaluate a checkpoint SuperGLUE:

`python eval.py name={pretrain name} eval=SuperGLUE impl.microbatch_size=16 seed=0 [impl.mixed_precision_target_dtype=bfloat16 if the checkpoint was trained using BF16 rather than FP16]`
