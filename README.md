# Open-Sora-Plan

[[Project Page]]()

![The architecture of Open-Sora-Plan](assets/framework.jpg)

This project aim to reproducing Sora (Open AI T2V model), but we only have limited resource.

The goal is to create a simple and scalable repo, to reproduce MUSE and build knowedge about VideoVAE + DiT at scale. We will use more data and GPUs for training.  We deeply wish the all open source community can contribute to this project.

这个项目旨在复现OpenAI的文生视频模型，但是我们计算资源有限，先搭建一个框架，我们希望整个开源社区能够一起合作共同为这个Project贡献力量，我们也会标注清楚大家的贡献。

## News

## Todo

- [x] support variable aspect ratios, resolutions, durations training
- [x] add class-conditioning on embeddings

- [ ] incorporating more conditions
- [ ] training with more data and more GPU


## Requirements and Installation

We recommend the requirements as follows.

* Python >= 3.8
* Pytorch >= 1.13.1
* CUDA Version >= 11.7
* Install required packages:

```
git clone https://github.com/PKU-YuanGroup/Open-Sora-Plan
cd Open-Sora-Plan
conda create -n opensora python=3.8 -y
conda activate opensora
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
cd VideoGPT
pip install -e .
cd ..
```

## Training

### VideoVAE

```
```

### VideoDiT
```

```

<p align="center">
<img src="assets/loss.jpg" width=50%>
</p>

## Acknowledgement
* [DiT](https://github.com/facebookresearch/DiT/tree/main) Scalable Diffusion Models with Transformers.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT) Video Generation using VQ-VAE and Transformers.
* [FiT](https://github.com/whlzy/Fi) Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595) Extending Context Window of Large Language Models via Positional Interpolation.

## License
* The service is a research preview intended for non-commercial use only. See [LICENSE.txt](LICENSE.txt) for details.

## Contributors

<a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/Open-Sora-Plan" />
</a>
