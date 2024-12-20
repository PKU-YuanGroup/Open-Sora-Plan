The original code is from [GenEval](https://github.com/djghosh13/geneval).

## Requirements and Installation

### Prepare the environment

> Official environment is not recommended which is for cuda 11.3.

Prepare conda environment:

```bash
conda create -n geneval_eval python=3.10
```

Activate the environment:

```bash
conda activate geneval_eval
```

Install torch:

```bash
conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
# conda install -c conda-forge nvcc_linux-64 # For compile mmcv (😇)
```

Install the requirements:

```bash
pip install -r requirements.txt
mim install mmengine mmcv-full==1.7.2
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

Third, download the Mask2Former object detection config and weights:

```bash
bash download_models.sh detector/
```

## Generate samples

Change back to conda `opensora` environment and run the following command to generate samples:

```bash
bash step1_gen_samples.sh
```

## Evaluation

```bash
bash step2_run_geneval.sh
```



## Problem

Due to bad MMCV compatibility, the code `evaluate_images.py` temporarily uses CPU inference.

## TODO

- [ ] Fix the problem of MMCV compatibility.
