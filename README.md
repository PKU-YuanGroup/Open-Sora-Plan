

<h1 align="left"> <a href="">Open-Sora Plan</a></h1>

This project aims to create a simple and scalable repo, to reproduce [Sora](https://openai.com/sora) (OpenAI, but we prefer to call it "ClosedAI" ). We wish the open-source community can contribute to this project. Pull requests are welcome! The current code supports complete training and inference using the Huawei Ascend AI computing system. Models trained on Huawei Ascend can also output video quality comparable to industry standards.

本项目希望通过开源社区的力量复现Sora，由北大-兔展AIGC联合实验室共同发起，当前版本离目标差距仍然较大，仍需持续完善和快速迭代，欢迎Pull request！目前代码同时支持使用国产AI计算系统（华为昇腾）进行完整的训练和推理。基于昇腾训练出的模型，也可输出持平业界的视频质量。

我们正在快速迭代新版本，欢迎更多合作者或算法工程师加入，[算法工程师招聘-兔展智能.pdf](https://github.com/user-attachments/files/19107972/-.pdf)

<h5 align="left">

[![arXiv](https://img.shields.io/badge/Arxiv-Open--Sora%20Plan-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.00131)
[![arXiv](https://img.shields.io/badge/Arxiv-WF--VAE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.17459)
[![License](https://img.shields.io/badge/License-Apache-yellow)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/LICENSE)  <br>
[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/DFZg5678)
[![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/issues/53#issuecomment-1987226516)
[![Twitter](https://img.shields.io/badge/-Twitter@LinBin46984-black?logo=twitter&logoColor=1D9BF0)](https://x.com/LinBin46984/status/1795018003345510687) 
[![Modelers](https://img.shields.io/badge/%E9%AD%94%E4%B9%90-%E6%A8%A1%E5%9E%8B%E4%BD%93%E9%AA%8C-blue)](https://modelers.cn/spaces/MindSpore-Lab/Open_Sora_Plan) <br>
[![GitHub repo stars](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/stargazers)&#160;
[![GitHub repo forks](https://img.shields.io/github/forks/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Forks)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/network)&#160;
[![GitHub repo watchers](https://img.shields.io/github/watchers/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Watchers)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/watchers)&#160;
[![GitHub repo size](https://img.shields.io/github/repo-size/PKU-YuanGroup/Open-Sora-Plan?style=flat&logo=github&logoColor=whitesmoke&label=Repo%20Size)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/archive/refs/heads/main.zip) <br>
[![GitHub repo contributors](https://img.shields.io/github/contributors-anon/PKU-YuanGroup/Open-Sora-Plan?style=flat&label=Contributors)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors) 
[![GitHub Commit](https://img.shields.io/github/commit-activity/m/PKU-YuanGroup/Open-Sora-Plan?label=Commit)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/commits/main/)
[![Pr](https://img.shields.io/github/issues-pr-closed-raw/PKU-YuanGroup/Open-Sora-Plan.svg?label=Merged+PRs&color=green)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/pulls)
[![GitHub issues](https://img.shields.io/github/issues/PKU-YuanGroup/Open-Sora-Plan?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/PKU-YuanGroup/Open-Sora-Plan?color=success&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aissue+is%3Aclosed)
</h5>
<a href="https://trendshift.io/repositories/8280" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8280" alt="PKU-YuanGroup%2FOpen-Sora-Plan | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
<h5 align="left"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


# 📣 News

* **[2025.06.06]** 🔥🔥🔥 We release version 1.5.0, Our most powerful model! By introducing a **higher-compression WFVAE** and an improved sparse DiT architecture, **SUV**, we achieve performance **comparable to HunyuanVideo (Open-Source)** using an 8B-scale model and 40 million video samples. Version 1.5.0 is **fully trained and inferred on Ascend 910-series accelerators**; Please check the [mindspeed_mmdit](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/mindspeed_mmdit) branch for our new code and [Report-v1.5.0.md](docs/Report-v1.5.0.md) for our report. The GPU version is coming soon. 
* **[2024.12.03]** ⚡️ We released our [arxiv paper](https://arxiv.org/abs/2412.00131) and WF-VAE [paper](https://arxiv.org/abs/2411.17459) for v1.3. The next more powerful version is coming soon.
* **[2024.10.16]** 🎉 We released version 1.3.0, featuring: **WFVAE**, **prompt refiner**, **data filtering strategy**, **sparse attention**, and **bucket training strategy**. We also support 93x480p within **24G VRAM**. More details can be found at our latest [report](docs/Report-v1.3.0.md).
* **[2024.08.13]** 🎉 We are launching Open-Sora Plan v1.2.0 **I2V** model, which is based on Open-Sora Plan v1.2.0. The current version supports image-to-video generation and transition generation (the starting and ending frames conditions for video generation). Check out the Image-to-Video section in this [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.2.0.md#training-image-to-video-diffusion-model).
* **[2024.07.24]** 🔥🔥🔥 v1.2.0 is here! Utilizing a 3D full attention architecture instead of 2+1D. We released a true 3D video diffusion model trained on 4s 720p. Check out our latest [report](docs/Report-v1.2.0.md).
* **[2024.05.27]** 🎉 We are launching Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out our latest [report](docs/Report-v1.1.0.md). Thanks to [ShareGPT4Video's](https://sharegpt4video.github.io/) capability to annotate long videos.
* **[2024.04.09]** 🤝 Excited to share our latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), which learns real-world physics knowledge from time-lapse videos.
* **[2024.04.07]** 🎉🎉🎉 Today, we are thrilled to present Open-Sora-Plan v1.0.0, which significantly enhances video generation quality and text control capabilities. See our [report](docs/Report-v1.0.0.md). Thanks to HUAWEI NPU for supporting us.
* **[2024.03.27]** 🚀🚀🚀 We release the report of [VideoCausalVAE](docs/CausalVideoVAE.md), which supports both images and videos. We present our reconstructed video in this demonstration as follows. The text-to-video model is on the way.
* **[2024.03.01]** 🤗 We launched a plan to reproduce Sora, called Open-Sora Plan! Welcome to **watch** 👀 this repository for the latest updates.

# 😍 Gallery

Text-to-Video Generation of Open-Sora Plan v1.5.0.
### Youtube:
[![Demo Video of Open-Sora Plan V1.5.0](https://github.com/user-attachments/assets/130bbba2-3ded-4092-92ef-b65b673cb1a6)](https://youtu.be/IiWTdx2EHCY)
### Bilibili:
[![Demo Video of Open-Sora Plan V1.5.0](https://github.com/user-attachments/assets/130bbba2-3ded-4092-92ef-b65b673cb1a6)](https://www.bilibili.com/video/BV1X77tzxE3b/)

# 😮 Highlights

Open-Sora Plan shows excellent performance in video generation.

### 🔥 WFVAE with Higher performance and compression
- With an 8×8×8 downsampling rate, but achieves higher PSNR than the VAE used in Wan2.1. Lowers the training cost for the DiT built upon it.

### 🚀 More powerful sparse dit
- The more powerful sparse attention architecture, SUV, achieves performance close to dense DiT while providing over a 35% speedup.

<p align="center">
    <img src="https://s21.ax1x.com/2024/07/22/pk7cob8.png" width="650" style="margin-bottom: 0.2;"/>
<p>

# 🐳 Resource

| Version | Architecture |  Diffusion Model | CausalVideoVAE | Data | Prompt Refiner |
|:---|:---|:---|:---|:---|:---|
| v1.5.0 | SUV (Skiparse 3D) | [121x576x1024](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0)[5] | [Anysize_8x8x8_32dim](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0) | - | - |
| v1.3.0 [4] | Skiparse 3D | [Anysize in 93x640x640](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/any93x640x640)[3], [Anysize in 93x640x640_i2v](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/any93x640x640_i2v)[3] | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/vae)| [prompt_refiner](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/prompt_refiner) | [checkpoint](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/prompt_refiner)| |
| v1.2.0 | Dense 3D | [93x720p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x720p), [29x720p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x720p)[1], [93x480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x480p)[1,2], [29x480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x480p), [1x480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/1x480p), [93x480p_i2v](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x480p_i2v) | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae)| [Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) | - |
| v1.1.0 | 2+1D | [221x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/221x512x512), [65x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/65x512x512) |[Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/vae) |[Data and Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0)| - |
| v1.0.0 | 2+1D | [65x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/65x512x512), [65x256x256](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/65x256x256), [17x256x256](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/17x256x256) | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/vae) | [Data and Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.0.0)| - |

> [1] Please note that the weights for v1.2.0 29×720p and 93×480p were trained on Panda70M and have not undergone final high-quality data fine-tuning, so they may produce watermarks.

> [2] We fine-tuned 3.5k steps from 93×720p to get 93×480p for community research use.

> [3] The model is trained arbitrarily on stride=32. So keep the resolution of the inference a multiple of 32. Frames need to be 4n+1, e.g. 93, 77, 61, 45, 29, 1 (image).

> [4] Model weights are also available at [OpenMind](https://modelers.cn/models/linbin/Open-Sora-Plan-v1.3.0) and [WiseModel](https://wisemodel.cn/models/PKU-YUAN/Open-Sora-Plan-v1.3.0).

> [5] The current model weights are only compatible with the NPU + MindSpeed-MM framework. Model weights are also available at and [modelers](https://modelers.cn/models/PKU-YUAN-Group/Open-Sora-Plan-v1.5.0).

> [!Warning]
>
> <div align="left">
> <b>
> 🚨 For version 1.2.0, we no longer support 2+1D models.
> </b>
> </div>

# ⚙️ How to start

### GPU
coming soon...
### NPU
Please check out the **[mindspeed_mmdit](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/mindspeed_mmdit)** branch and follow the README.md for configuration.

# 📖 Technical report
Please check [Report-v1.5.0.md](docs/Report-v1.5.0.md).

# 💡 How to Contribute
We greatly appreciate your contributions to the Open-Sora Plan open-source community and helping us make it even better than it is now!

For more details, please refer to the [Contribution Guidelines](docs/Contribution_Guidelines.md)

# 👍 Acknowledgement and Related Work
* [Allegro](https://github.com/rhymes-ai/Allegro): Allegro is a powerful text-to-video model that generates high-quality videos up to 6 seconds at 15 FPS and 720p resolution from simple text input based on our Open-Sora Plan. The significance of open-source is becoming increasingly tangible.
* [Latte](https://github.com/Vchitect/Latte): It is a wonderful 2+1D video generation model.
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


```bibtex
@article{lin2024open,
  title={Open-Sora Plan: Open-Source Large Video Generation Model},
  author={Lin, Bin and Ge, Yunyang and Cheng, Xinhua and Li, Zongjian and Zhu, Bin and Wang, Shaodong and He, Xianyi and Ye, Yang and Yuan, Shenghai and Chen, Liuhan and others},
  journal={arXiv preprint arXiv:2412.00131},
  year={2024}
}
```

```bibtex
@article{li2024wf,
  title={WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model},
  author={Li, Zongjian and Lin, Bin and Ye, Yang and Chen, Liuhan and Cheng, Xinhua and Yuan, Shenghai and Yuan, Li},
  journal={arXiv preprint arXiv:2411.17459},
  year={2024}
}
```

# 🤝 Community contributors

<a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/Open-Sora-Plan" />
</a>

