# Open-Sora-Plan
This project aim to reproducing Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project.
这个项目旨在复现OpenAI的文生视频模型，但是我们计算资源有限，先搭建一个框架，我们希望整个开源社区能够一起合作共同为这个Project贡献力量，我们也会标注清楚大家的贡献。

## VideoVAE

### Install
```
conda create -n videovae python=3.8 -y
conda activate videovae
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.3.1 kornia==0.6.4 torchmetrics==0.5.0
pip install opencv-python==4.1.2.30 pytorch-lightning==1.4.2 omegaconf==2.1.1 einops==0.3.0 decord
pip install pytorchvideo six test-tube albumentations
pip install timm accelerate torchdiffeq
cd latent-diffusion
pip install -e .
cd ..
cd taming-transformers
pip install -e .
cd ..
```

### Inference (VideoVae)
Download checkpoint and specify the path in `demo_videovae.py`

```
python demo_videovae.py
```

### Inference (SiT_dynamic)

```
cd SiT_dynamic
# for image
python sample_videovae.py ODE --image-size 256 --seed 1 --is-image True
# for video
python sample_videovae.py ODE --image-size 256 --seed 1 --is-image False
```
