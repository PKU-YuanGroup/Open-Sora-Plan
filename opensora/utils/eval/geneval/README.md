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
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c nvidia cudatoolkit=11.8
conda install -c conda-forge nvcc_linux-64 # For compile mmcv (😇)
```

Install the MMCV:


**H100**
```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv

git checkout v2.2.0
pip install -r requirements/optional.txt
vim setup.py
# L160
extra_compile_args = {
    # 'nvcc': [cuda_args, '-std=c++14'] if cuda_args else ['-std=c++14'],
    'nvcc': [cuda_args, '-std=c++14', '-arch=sm_90'] if cuda_args else ['-std=c++14'],
    'cxx': ['-std=c++14'],
}
# Revert all changes to setup.py using Ctrl+Z. Then, Ctrl+S to save
pip install -v -e .
git checkout v1.7.0
vim setup.py
# L217
extra_compile_args = {
    # 'nvcc': [cuda_args, '-std=c++14'] if cuda_args else ['-std=c++14'],
    'nvcc': [cuda_args, '-std=c++14', '-arch=sm_90'] if cuda_args else ['-std=c++14'],
    'cxx': ['-std=c++14'],
}
pip install -v -e .
python .dev_scripts/check_installation.py
cd ..
```

**Other GPU**
```
mim install mmengine mmcv-full==1.7.0
```

Install the MMDet:

```bash
# pip install -r requirements.txt
# mim install mmengine mmcv-full==1.7.2
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

Download the Mask2Former object detection config and weights:

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
bash step2_eval_samples.sh
```

## Summary the scores   

```bash
bash step3_summary_score.sh
```
