

<h1 align="left"> <a href="">Open-Sora Plan</a></h1>

This project aims to create a simple and scalable repo, to reproduce [Sora](https://openai.com/sora) (OpenAI, but we prefer to call it "ClosedAI" ). We wish the open-source community can contribute to this project. Pull requests are welcome! The current code supports complete training and inference using the Huawei Ascend AI computing system. Models trained on Huawei Ascend can also output video quality comparable to industry standards.

本项目希望通过开源社区的力量复现Sora，由北大-兔展AIGC联合实验室共同发起，当前版本离目标差距仍然较大，仍需持续完善和快速迭代，欢迎Pull request！目前代码同时支持使用国产AI计算系统（华为昇腾）进行完整的训练和推理。基于昇腾训练出的模型，也可输出持平业界的视频质量。

<h5 align="left">

[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/DFZg5678)
[![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/issues/53#issuecomment-1987226516)
[![Twitter](https://img.shields.io/badge/-Twitter@LinBin46984-black?logo=twitter&logoColor=1D9BF0)](https://x.com/LinBin46984/status/1795018003345510687) <br>
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/LICENSE) 
[![GitHub repo contributors](https://img.shields.io/github/contributors-anon/PKU-YuanGroup/Open-Sora-Plan?style=flat&label=Contributors)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors) 
[![GitHub Commit](https://img.shields.io/github/commit-activity/m/PKU-YuanGroup/Open-Sora-Plan?label=Commit)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/commits/main/)
[![Pr](https://img.shields.io/github/issues-pr-closed-raw/PKU-YuanGroup/Open-Sora-Plan.svg?label=Merged+PRs&color=green)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/pulls)
[![GitHub issues](https://img.shields.io/github/issues/PKU-YuanGroup/Open-Sora-Plan?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/PKU-YuanGroup/Open-Sora-Plan?color=success&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aissue+is%3Aclosed) <br>
[![GitHub repo stars](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/stargazers)&#160;
[![GitHub repo forks](https://img.shields.io/github/forks/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Forks)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/network)&#160;
[![GitHub repo watchers](https://img.shields.io/github/watchers/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Watchers)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/watchers)&#160;
[![GitHub repo size](https://img.shields.io/github/repo-size/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Repo%20Size)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/archive/refs/heads/main.zip)
</h5>
<h5 align="left"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


# 📣 News
* `COMING SOON` ⚡️⚡️⚡️ For large model parallelisation training, TP & SP and more strategies are coming...
  
  > 近期将新增华为昇腾多模态MindSpeed-MM分支，借助华为MindSpeed-MM套件的能力支撑Open-Sora Plan参数的扩增，为更大参数规模的模型训练提供TP、SP等分布式训练能力。

* **[2024.10.16]** 🎉 We released version 1.3.0, featuring: **WFVAE**, **pompt refiner**, **data filtering strategy**, **sparse attention**, and **bucket training strategy**. We also support 93x480p within **24G VRAM**. More details can be found at our latest [report](docs/Report-v1.3.0.md).
* **[2024.08.13]** 🎉 We are launching Open-Sora Plan v1.2.0 **I2V** model, which based on Open-Sora Plan v1.2.0. The current version supports image-to-video generation and transition generation (the starting and ending frames conditions for video generation). Checking out the Image-to-Video section in this [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.2.0.md#training-image-to-video-diffusion-model).
* **[2024.07.24]** 🔥🔥🔥 v1.2.0 is here! Utilizing a 3D full attention architecture instead of 2+1D. We released a true 3D video diffusion model trained on 4s 720p. Checking out our latest [report](docs/Report-v1.2.0.md).
* **[2024.05.27]** 🎉 We are launching Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out our latest [report](docs/Report-v1.1.0.md). Thanks to [ShareGPT4Video's](https://sharegpt4video.github.io/) capability to annotate long videos.
* **[2024.04.09]** 🤝 Excited to share our latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), which learns real-world physics knowledge from time-lapse videos.
* **[2024.04.07]** 🎉🎉🎉 Today, we are thrilled to present Open-Sora-Plan v1.0.0, which significantly enhances video generation quality and text control capabilities. See our [report](docs/Report-v1.0.0.md). Thanks to HUAWEI NPU for supporting us.
* **[2024.03.27]** 🚀🚀🚀 We release the report of [VideoCausalVAE](docs/CausalVideoVAE.md), which supports both images and videos. We present our reconstructed video in this demonstration as follows. The text-to-video model is on the way.
* **[2024.03.01]** 🤗 We launched a plan to reproduce Sora, called Open-Sora Plan! Welcome to **watch** 👀 this repository for the latest updates.

# 😍 Gallery

Text & Image to Video Generation. 

[![Demo Video of Open-Sora Plan V1.3](https://github.com/user-attachments/assets/4ff1d873-3dde-4905-a907-dbff51174c20)](https://www.bilibili.com/video/BV1KR2fYPEF5/?spm_id_from=333.999.0.0&vd_source=cfda99203e659100629b465161f1d87d)

# 😮 Highlights

Open-Sora Plan shows excellent performance in video generation.

### 🔥 High performance CausalVideoVAE, but with fewer training cost
- High compression ratio with excellent performance, capable of **compressing videos by 256 times (4×8×8)**. Causal convolution supports simultaneous inference of images and videos but only need **1 node to train**.

### 🚀 Video Diffusion Model based on 3D attention, joint learning of spatiotemporal features.
- With **a new sparse attention architecture** instead of a 2+1D model, 3D attention can better capture joint spatial and temporal features.

<p align="center">
    <img src="https://s21.ax1x.com/2024/07/22/pk7cob8.png" width="650" style="margin-bottom: 0.2;"/>
<p>

# 🤗 Demo

### Gradio Web UI

Highly recommend trying out our web demo by the following command.

```bash
python -m opensora.serve.gradio_web_server --model_path "path/to/model" \
    --ae WFVAEModel_D8_4x8x8 --ae_path "path/to/vae" \
    --caption_refiner "path/to/refiner" \
    --text_encoder_name_1 "path/to/text_enc" --rescale_betas_zero_snr
```

### ComfyUI

Coming soon...

# 🐳 Resource

| Version | Architecture |  Diffusion Model | CausalVideoVAE | Data | Prompt Refiner |
|:---|:---|:---|:---|:---|:---|
| v1.3.0 | 3D | [Anysize in 93x640x640](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/any93x640x640)[3], more checkpoints are coming soon | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/vae)| - | [checkpoint](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/prompt_refiner)| |
| v1.2.0 | 3D | [93x720p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x720p), [29x720p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x720p)[1], [93x480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x480p)[1,2], [29x480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x480p), [1x480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/1x480p), [93x480p_i2v](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x480p_i2v) | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae)| [Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) | - |
| v1.1.0 | 2+1D | [221x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/221x512x512), [65x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/65x512x512) |[Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/vae) |[Data and Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0)| - |
| v1.0.0 | 2+1D | [65x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/65x512x512), [65x256x256](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/65x256x256), [17x256x256](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/17x256x256) | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/vae) | [Data and Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.0.0)| - |

> [1] Please note that the weights for v1.2.0 29×720p and 93×480p were trained on Panda70M and have not undergone final high-quality data fine-tuning, so they may produce watermarks.

> [2] We fine-tuned 3.5k steps from 93×720p to get 93×480p for community research use.

> [3] The model is trained arbitrarily on stride=32. So keep the resolution of the inference a multiple of 32. Frames needs to be 4n+1, e.g. 93, 77, 61, 45, 29, 1 (image).

> [!Warning]
>
> <div align="left">
> <b>
> 🚨 For version 1.2.0, we no longer support 2+1D models.
> </b>
> </div>

# ⚙️ Requirements and Installation

1. Clone this repository and navigate to Open-Sora-Plan folder
```
git clone https://github.com/PKU-YuanGroup/Open-Sora-Plan
cd Open-Sora-Plan
```
2. Install required packages
We recommend the requirements as follows.
* Python >= 3.8
* Pytorch >= 2.1.0
* CUDA Version >= 11.7
```
conda create -n opensora python=3.8 -y
conda activate opensora
pip install -e .
```

3. Install optional requirements such as static type checking:
```
pip install -e '.[dev]'
```

# 🗝️ Training & Validating

## 🗜️ CausalVideoVAE

The data preparation, training, inferencing and evaluation can be found [here](docs/VAE.md)

## 📜 Text-to-Video 

The data preparation, training and inferencing can be found [here](docs/T2V.md)

## 🖼️ Image-to-Video

The data preparation, training and inferencing can be found [here](docs/I2V.md)


# ⚡️ Extra Save Memory

## 🔆 Training
During training, the entire EMA model remains in VRAM. You can enable `--offload_ema` or disable `--use_ema`. Additionally, VAE tiling is disabled by default, but you can pass `--enable_tiling` or disable `--vae_fp32`. Finally, a temporary but extreme saving memory option is enable `--extra_save_mem` to offload the text encoder and VAE to the CPU when not in use, though this will significantly slow down performance.

We currently have two plans: one is to continue using the Deepspeed/FSDP approach, sharding the EMA and text encoder across ranks with Zero3, which is sufficient for training 10-15B models. The other is to adopt MindSpeed for various parallel strategies, enabling us to scale the model up to 30B.

## ⚡️ 24G VRAM Inferencing

Please first ensure that you understand how to inference. Refer to the [inference](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/T2V.md#inference) instructions in Text-to-Video.
Simply specify `--save_memory`, and during inference, `enable_model_cpu_offload()`, `enable_sequential_cpu_offload()`, and `vae.vae.enable_tiling()` will be automatically activated.

# 💡 How to Contribute
We greatly appreciate your contributions to the Open-Sora Plan open-source community and helping us make it even better than it is now!

For more details, please refer to the [Contribution Guidelines](docs/Contribution_Guidelines.md)

# 👍 Acknowledgement
* [Latte](https://github.com/Vchitect/Latte): It is an wonderful 2+1D video generated model.
* [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.
* [ShareGPT4Video](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4Video): Improving Video Understanding and Generation with Better Captions.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT): Video Generation using VQ-VAE and Transformers.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [FiT](https://github.com/whlzy/FiT): Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595): Extending Context Window of Large Language Models via Positional Interpolation.


# 🔒 License
* See [LICENSE](LICENSE) for details.

## ✨ Star History

[![Star History](https://api.star-history.com/svg?repos=PKU-YuanGroup/Open-Sora-Plan)](https://star-history.com/#PKU-YuanGroup/Open-Sora-Plan&Date)



# ✏️ Citing

## BibTeX

```bibtex
@software{pku_yuan_lab_and_tuzhan_ai_etc_2024_10948109,
  author       = {PKU-Yuan Lab and Tuzhan AI etc.},
  title        = {Open-Sora-Plan},
  month        = apr,
  year         = 2024,
  publisher    = {GitHub},
  doi          = {10.5281/zenodo.10948109},
  url          = {https://doi.org/10.5281/zenodo.10948109}
}
```
## Latest DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10948109.svg)](https://zenodo.org/records/10948109)

# 🤝 Community contributors

<a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/Open-Sora-Plan" />
</a>

