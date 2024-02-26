# Open-Sora-Plan

This project aim to reproducing Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project.

这个项目旨在复现OpenAI的文生视频模型，但是我们计算资源有限，先搭建一个框架，我们希望整个开源社区能够一起合作共同为这个Project贡献力量，我们也会标注清楚大家的贡献。

## Install
```
conda create -n opensora python=3.8 -y
conda activate opensora
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.32.0 kornia==0.6.4 torchmetrics==0.5.0
pip install opencv-python==4.1.2.30 pytorch-lightning==1.4.2 omegaconf==2.1.1 einops==0.3.0
pip install pytorchvideo six test-tube albumentations timm accelerate torchdiffeq wandb decord
cd latent-diffusion
pip install -e .
cd ..
cd taming-transformers
pip install -e .
cd ..
```

## Training

### VideoVAE

```
cd latent-diffusion
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base configs/autoencoder/ucf_ae_kl_16x16x4.yaml -t --gpus 0,1,2,3,4,5,6,7,
```

### VideoSiT
```
cd SiT
export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
export ENTITY="your_wandb_name"
export PROJECT="your_proj_name"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
    --model SiT-XL/2 --ckpt pretrained_models/SiT-XL-2-256x256.pt \
	--data-path /remote-home/yeyang/UCF-101 --num-classes 1000 \
	--sample-rate 4 --num-frames 16 --temproal-size 1 \
	--max-image-size 256 --epochs 200 --global-batch-size 128 --lr 1e-4 \
	--ckpt-every 1000 --sample-every 1000 --wandb
```
