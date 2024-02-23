## Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (SiT)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/pdf/2401.08740.pdf) | [Project Page](https://scalable-interpolant.github.io/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/willisma/SiT/blob/main/run_SiT.ipynb)

![SiT samples](visuals/visual.png)

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper exploring 
interpolant models with scalable transformers (SiTs). 

> [**Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**](https://arxiv.org/pdf/2401.08740.pdf)<br>
> [Nanye Ma](https://willisma.github.io), [Mark Goldstein](https://marikgoldstein.github.io/), [Michael Albergo](http://malbergo.me/), [Nicholas Boffi](https://nmboffi.github.io/), [Eric Vanden-Eijnden](https://wp.nyu.edu/courantinstituteofmathematicalsciences-eve2/), [Saining Xie](https://www.sainingxie.com)
> <br>New York University<br>

We present Scalable Interpolant Transformers (SiT), a family of generative models built on the backbone of Diffusion Transformers (DiT). The interpolant framework, which allows for connecting two distributions in a more flexible way than standard diffusion models, makes possible a modular study of various design choices impacting generative models built on dynamical transport: using discrete vs. continuous time learning, deciding the model to learn, choosing the interpolant connecting the distributions, and deploying a deterministic or stochastic sampler. By carefully introducing the above ingredients, SiT surpasses DiT uniformly across model sizes on the conditional ImageNet 256x256 benchmark using the exact same backbone, number of parameters, and GFLOPs. By exploring various diffusion coefficients, which can be tuned separately from learning, SiT achieves an FID-50K score of 2.06.

This repository contains:

* 🪐 A simple PyTorch [implementation](models.py) of SiT
* ⚡️ Pre-trained class-conditional SiT models trained on ImageNet 256x256
* 🛸 A SiT [training script](train.py) using PyTorch DDP

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/willisma/SiT.git
cd SiT
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate SiT
```


## Sampling [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/willisma/SiT/blob/main/run_SiT.ipynb)
![More SiT samples](visuals/visual_2.png)

**Pre-trained SiT checkpoints.** You can sample from our pre-trained SiT models with [`sample.py`](sample.py). Weights for our pre-trained SiT model will be 
automatically downloaded depending on the model you use. The script has various arguments to adjust sampler configurations (ODE & SDE), sampling steps, change the classifier-free guidance scale, etc. For example, to sample from
our 256x256 SiT-XL model with default ODE setting, you can use:

```bash
python sample.py ODE --image-size 256 --seed 1
```

For convenience, our pre-trained SiT models can be downloaded directly here as well:

| SiT Model     | Image Resolution | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) | 256x256          | 2.06    | 270.27         | 119    |
<!-- | [XL/2](https://dl.fbaipublicfiles.com/SiT/models/SiT-XL-2-512x512.pt) | 512x512          | 3.04    | 240.82          | 525    | -->


**Custom SiT checkpoints.** If you've trained a new SiT model with [`train.py`](train.py) (see [below](#training-SiT)), you can add the `--ckpt`
argument to use your own checkpoint instead. For example, to sample from the EMA weights of a custom 
256x256 SiT-L/4 model with ODE sampler, run:

```bash
python sample.py ODE --model SiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```

### Advanced sampler settings

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| ODE | `--atol` | `float` |  Absolute error tolerance |
|     | `--rtol` | `float` | Relative error tolenrace |   
|     | `--sampling-method` | `str` | Sampling methods (refer to [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) ) |

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| SDE | `--diffusion-form` | `str` | Form of SDE's diffusion coefficient (refer to Tab. 2 in [paper]()) |
|     | `--diffusion-norm` | `float` | Magnitude of SDE's diffusion coefficient |
|     | `--last-step` | `str` | Form of SDE's last step |
|     |               |       | None - Single SDE integration step |
|     |               |       | "Mean" - SDE integration step without diffusion coefficient |
|     |               |       | "Tweedie" - [Tweedie's denoising](https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf) step | 
|     |               |       | "Euler" - Single ODE integration step
|     | `--sampling-method` | `str` | Sampling methods |
|     |               |       | "Euler" - First order integration | 
|     |               |       | "Heun" - Second order integration | 

There are some more options; refer to [`train_utils.py`](train_utils.py) for details.

## Training SiT

We provide a training script for SiT in [`train.py`](train.py). To launch SiT-XL/2 (256x256) training with `N` GPUs on 
one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train
```

**Logging.** To enable `wandb`, firstly set `WANDB_KEY`, `ENTITY`, and `PROJECT` as environment variables:

```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```

Then in training command add the `--wandb` flag:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --wandb
```

**Interpolant settings.** We also support different choices of interpolant and model predictions. For example, to launch SiT-XL/2 (256x256) with `Linear` interpolant and `noise` prediction: 

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --path-type Linear --prediction noise
```

**Resume training.** To resume training from custom checkpoint:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-L/2 --data-path /path/to/imagenet/train --ckpt /path/to/model.pt
```

**Caution.** Resuming training will automatically restore both model, EMA, and optimizer states and training configs to be the same as in the checkpoint.

<!-- ### PyTorch Training Results

We've trained SiT-XL/2 and SiT-B/2 models from scratch with the PyTorch training script
to verify that it reproduces the original JAX results up to several hundred thousand training iterations. Across our experiments, the PyTorch-trained models give 
similar (and sometimes slightly better) results compared to the JAX-trained models up to reasonable random variation. Some data points:

| SiT Model  | Train Steps | FID-50K<br> (JAX Training) | FID-50K<br> (PyTorch Training) | PyTorch Global Training Seed |
|------------|-------------|----------------------------|--------------------------------|------------------------------|
| XL/2       | 400K        | 19.5                       | **18.1**                       | 42                           |
| B/4        | 400K        | **68.4**                   | 68.9                           | 42                           |
| B/4        | 400K        | 68.4                       | **68.3**                       | 100                          |

These models were trained at 256x256 resolution; we used 8x A100s to train XL/2 and 4x A100s to train B/4. Note that FID 
here is computed with 250 DDPM sampling steps, with the `mse` VAE decoder and without guidance (`cfg-scale=1`). 

**TF32 Note (important for A100 users).** When we ran the above tests, TF32 matmuls were disabled per PyTorch's defaults. 
We've enabled them at the top of `train.py` and `sample.py` because it makes training and sampling way way way faster on 
A100s (and should for other Ampere GPUs too), but note that the use of TF32 may lead to some differences compared to 
the above results. -->

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a SiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained SiT-XL/2 model over `N` GPUs under default ODE sampler settings, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 50000
```

**Likelihood.** Likelihood evaluation is supported. To calculate likelihood, you can add the `--likelihood` flag to ODE sampler:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --likelihood
```

Notice that only under ODE sampler likelihood can be calculated; see [`sample_ddp.py`](sample_ddp.py) for more details and settings. 

### Enhancements
Training (and sampling) could likely be speed-up significantly by:
- [ ] using [Flash Attention](https://github.com/HazyResearch/flash-attention) in the SiT model
- [ ] using `torch.compile` in PyTorch 2.0

Basic features that would be nice to add:
- [ ] Monitor FID and other metrics
- [ ] AMP/bfloat16 support

Precision in likelihood calculation could likely be improved by:
- [ ] Uniform / Gaussian Dequantization


## Differences from JAX

Our models were originally trained in JAX on TPUs. The weights in this repo are ported directly from the JAX models. 
There may be minor differences in results stemming from sampling on different platforms (TPU vs. GPU). We observed that sampling on TPU performs marginally worse than GPU (2.15 FID 
versus 2.06 in the paper).


## License
This project is under the MIT license. See [LICENSE](LICENSE.txt) for details.


