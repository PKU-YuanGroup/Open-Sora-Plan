# Open-Sora Plan
<!--
[[Project Page]](https://pku-yuangroup.github.io/Open-Sora-Plan/) [[中文主页]](https://pku-yuangroup.github.io/Open-Sora-Plan/blog_cn.html)
-->


[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/vqGmpjkSaz)
[![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/issues/53#issuecomment-1987226516)
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.1.0)
[![Twitter](https://img.shields.io/badge/-Twitter@LinBin46984-black?logo=twitter&logoColor=1D9BF0)](https://x.com/LinBin46984/status/1763476690385424554?s=20) <br>
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.1.0)
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

<details>
<summary>v1.0.0 badge</summary>
[![Twitter](https://img.shields.io/badge/-Twitter@LinBin46984-black?logo=twitter&logoColor=1D9BF0)](https://x.com/LinBin46984/status/1763476690385424554?s=20) <br>
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0)
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/fffiloni/Open-Sora-Plan-v1-0-0)
[![Replicate demo and cloud API](https://replicate.com/camenduru/open-sora-plan-512x512/badge)](https://replicate.com/camenduru/open-sora-plan-512x512)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/Open-Sora-Plan-jupyter/blob/main/Open_Sora_Plan_jupyter.ipynb) <br>
</details>

We are thrilled to present **Open-Sora-Plan v1.1.0**, which significantly enhances video generation quality and text control capabilities. See our [report](docs/Report-v1.1.0.md). We show compressed .gif on GitHub, which loses some quality.

Thanks to **HUAWEI Ascend NPU Team** for supporting us.

目前已支持国产AI芯片(华为昇腾，期待更多国产算力芯片)进行完整地训练推理，具体可参考昇腾分支[hw branch](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/hw). 


### 221×512×512 Text-to-Video Generation



| 221×512×512 (9.2s) | 221×512×512 (9.2s) | 221×512×512 (9.2s) |
| --- | --- | --- |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/98897215-eae9-49f3-8fdb-df1d4b74d435" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/5453cf37-29ac-423d-9fb2-05f23416ca3e" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/77928824-6705-4d83-a7b4-43fdb74790bf" width=224> |
| 3D animation of a small, round, fluffy creature with big, expressive eyes explores ... |  A single drop of liquid metal falls from a floating orb, landing on a mirror-like ... | The video presents an abstract composition centered around a hexagonal shape adorned ...  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/42542b4e-b1b8-49b8-ada4-7bcfde8e1453" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/4eee2619-a5ca-4a32-b350-e65d6220c8f7" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/ec78d855-07c7-4421-896a-3c46a83ec129" width=224> |
| A drone camera circles around a beautiful historic church built on a rocky outcropping ... | Aerial view of Santorini during the blue hour, showcasing the stunning architecture ...  | An aerial shot of a lighthouse standing tall on a rocky cliff, its beacon cutting ...  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/f9c5823f-aa03-40ee-8335-684684f5c842" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/f207d798-8988-45b0-b836-347e499ee000" width=224>  |  <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/67b40ce1-135d-4ae2-a1ee-5a0866461eb2" width=224> |
| A snowy forest landscape with a dirt road running through it. The road is flanked by ... |  Drone shot along the Hawaii jungle coastline, sunny day. Kayaks in the water. |The camera rotates around a large stack of vintage televisions all showing different ...  |


### 65×512×512 Text-to-Video Generation

| 65×512×512 (2.7s) | 65×512×512 (2.7s) | 65×512×512 (2.7s) |
| --- | --- | --- |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/8edfa249-272c-4773-8728-12686527771e" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/33338d8e-5ea7-4b57-9e94-97f3ee404033" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/227fa6b6-801b-438b-9b30-cd3a4e0a7f2f" width=224> | 
| In an ornate, historical hall, a massive tidal wave peaks and begins to crash. Two ... | A Shiba Inu dog wearing a beret and black turtleneck.  |   A painting of a boat on water comes to life, with waves crashing and the boat becoming ...  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/a0601f20-579c-4e2e-832c-5763546718cc" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/55229eca-de3a-476b-930b-13a35eb5db30" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/66b06822-2652-453e-80fb-dd2988b730ce" width=224> |
| Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. | 3D animation of a small, round, fluffy creature with big, expressive eyes explores a ...   |    In a studio, there is a painting depicting a ship sailing through the rough sea. |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/4d27ff13-e725-4602-bf17-90df2c0d8005" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/aba01259-60f2-49ef-aa33-e738dc8c9a49" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/98178397-61b0-4200-9ab9-9bef2728ed98" width=224> |
| A robot dog trots down a deserted alley at night, its metallic paws clinking softly ... |   A lone surfer rides a massive wave, skillfully maneuvering through the surf. The water ... |  A solitary cheetah sprints across the savannah, its powerful muscles propelling it ... |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/3964dfcd-d1b4-406b-916c-d4a702184a27" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/34c1dda6-e420-4edd-bbcd-38ff8e542ec6" width=224>  |<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/c82b822b-cadd-4f11-bcc3-29e3651c02c0" width=224> |
| A solitary astronaut plants a flag on an alien planet covered in crystal formations ... | A person clad in a space suit with a helmet and equipped with a chest light and arm ... |  A wise old elephant painting abstract art with its trunk, each stroke a burst of color ... |
### 65×512×512 Video Editing

| generated 65×512×512 (2.7s) | edited 65×512×512 (2.7s) |
| --- | --- |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/edb8e8c2-5eef-4c90-85fb-6adb035067c3" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/32d93845-8904-4f8f-832e-37eba2ceb542" width=224>  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/a70d0d91-0d61-4aa4-9520-6e4a6c477f12" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/7913e06f-1e0b-4d06-8233-72c882c6abfe" width=224>  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/0e614ec0-fba0-4f42-a343-d4607966dd40" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/c531d930-410c-4614-890d-bae8013f33c2" width=224>  |

### 512×512 Text-to-Image Generation

<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/e44b7f8a-5da2-49c2-87c4-52ea680ad43b" width=512> 




## 📰 News

**[2024.05.27]** 🚀🚀🚀 We are launching Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out our latest [report](docs/Report-v1.1.0.md).

**[2024.04.09]** 🚀 Excited to share our latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), which learns real-world physics knowledge from time-lapse videos. Here is the dataset for train (updating): [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset).

**[2024.04.07]** 🔥🔥🔥 Today, we are thrilled to present Open-Sora-Plan v1.0.0, which significantly enhances video generation quality and text control capabilities. See our [report](docs/Report-v1.0.0.md). Thanks to HUAWEI NPU for supporting us.

**[2024.03.27]** 🚀🚀🚀 We release the report of [VideoCausalVAE](docs/CausalVideoVAE.md), which supports both images and videos. We present our reconstructed video in this demonstration as follows. The text-to-video model is on the way.

<details>
<summary>View more</summary>
  
**[2024.03.10]** 🚀🚀🚀 This repo supports training a latent size of 225×90×90 (t×h×w), which means we are able to **train 1 minute of 1080P video with 30FPS** (2× interpolated frames and 2× super resolution) under class-condition.

**[2024.03.08]** We support the training code of text condition with 16 frames of 512x512. The code is mainly borrowed from [Latte](https://github.com/Vchitect/Latte).

**[2024.03.07]** We support training with 128 frames (when sample rate = 3, which is about 13 seconds) of 256x256, or 64 frames (which is about 6 seconds) of 512x512.

**[2024.03.05]** See our latest [todo](https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file#todo), pull requests are welcome.

**[2024.03.04]** We re-organize and modulize our code to make it easy to [contribute](https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file#how-to-contribute-to-the-open-sora-plan-community) to the project, to contribute please see the [Repo structure](https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file#repo-structure).

**[2024.03.03]** We open some [discussions](https://github.com/PKU-YuanGroup/Open-Sora-Plan/discussions) to clarify several issues.

**[2024.03.01]** Training code is available now! Learn more on our [project page](https://pku-yuangroup.github.io/Open-Sora-Plan/). Please feel free to watch 👀 this repository for the latest updates.

</details>

## 💪 Goal
This project aims to create a simple and scalable repo, to reproduce [Sora](https://openai.com/sora) (OpenAI, but we prefer to call it "ClosedAI" ). We wish the open-source community can contribute to this project. Pull requests are welcome!!!

本项目希望通过开源社区的力量复现Sora，由北大-兔展AIGC联合实验室共同发起，当前版本离目标差距仍然较大，仍需持续完善和快速迭代，欢迎Pull request！！！

Project stages:
- Primary
1. Setup the codebase and train an un-conditional model on a landscape dataset.
2. Train models that boost resolution and duration.

- Extensions
3. Conduct text2video experiments on landscape dataset.
4. Train the 1080p model on video2text dataset.
5. Control model with more conditions.


<div style="display: flex; justify-content: center;"> 
  <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/6b3095e9-88e8-4481-9b1b-ff9aaa25caf1" width=200> 
  <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/f0a2ebca-6d25-4f94-be29-bd0a29cd9230" width=600> 
</div>

  
<details>
<summary>✊ Todo</summary>

#### Setup the codebase and train an unconditional model on landscape dataset
- [x] Fix typos & Update readme. 🤝 Thanks to [@mio2333](https://github.com/mio2333), [@CreamyLong](https://github.com/CreamyLong), [@chg0901](https://github.com/chg0901), [@Nyx-177](https://github.com/Nyx-177), [@HowardLi1984](https://github.com/HowardLi1984), [@sennnnn](https://github.com/sennnnn), [@Jason-fan20](https://github.com/Jason-fan20)
- [x] Setup environment. 🤝 Thanks to [@nameless1117](https://github.com/nameless1117)
- [ ] Add docker file. ⌛ [WIP] 🤝 Thanks to [@Mon-ius](https://github.com/Mon-ius), [@SimonLeeGit](https://github.com/SimonLeeGit)
- [ ] Enable type hints for functions. 🤝 Thanks to [@RuslanPeresy](https://github.com/RuslanPeresy), 🙏 **[Need your contribution]**
- [x] Resume from checkpoint.
- [x] Add Video-VQVAE model, which is borrowed from [VideoGPT](https://github.com/wilson1yan/VideoGPT).
- [x] Support variable aspect ratios, resolutions, durations training on [DiT](https://github.com/facebookresearch/DiT).
- [x] Support Dynamic mask input inspired by [FiT](https://github.com/whlzy/FiT).
- [x] Add class-conditioning on embeddings.
- [x] Incorporating [Latte](https://github.com/Vchitect/Latte) as main codebase.
- [x] Add VAE model, which is borrowed from [Stable Diffusion](https://github.com/CompVis/latent-diffusion).
- [x] Joint dynamic mask input with VAE.
- [ ] Add VQVAE from [VQGAN](https://github.com/CompVis/taming-transformers). 🙏 **[Need your contribution]**
- [ ] Make the codebase ready for the cluster training. Add SLURM scripts. 🙏 **[Need your contribution]**
- [x] Refactor VideoGPT. 🤝 Thanks to [@qqingzheng](https://github.com/qqingzheng), [@luo3300612](https://github.com/luo3300612), [@sennnnn](https://github.com/sennnnn)
- [x] Add sampling script.
- [ ] Add DDP sampling script. ⌛ [WIP]
- [x] Use accelerate on multi-node. 🤝 Thanks to [@sysuyy](https://github.com/sysuyy)
- [x] Incorporate [SiT](https://github.com/willisma/SiT). 🤝 Thanks to [@khan-yin](https://github.com/khan-yin)
- [x] Add evaluation scripts (FVD, CLIP score). 🤝 Thanks to [@rain305f](https://github.com/rain305f)

#### Train models that boost resolution and duration
- [x] Add [PI](https://arxiv.org/abs/2306.15595) to support out-of-domain size. 🤝 Thanks to [@jpthu17](https://github.com/jpthu17)
- [x] Add 2D RoPE to improve generalization ability as [FiT](https://github.com/whlzy/FiT). 🤝 Thanks to [@jpthu17](https://github.com/jpthu17)
- [x] Compress KV according to [PixArt-sigma](https://pixart-alpha.github.io/PixArt-sigma-project). 
- [x] Support deepspeed for videogpt training. 🤝 Thanks to [@sennnnn](https://github.com/sennnnn)
- [x] Train a **low dimension** Video-AE, whether it is VAE or VQVAE.
- [x] Extract offline feature.
- [x] Train with offline feature.
- [x] Add frame interpolation model. 🤝 Thanks to [@yunyangge](https://github.com/yunyangge)
- [x] Add super resolution model. 🤝 Thanks to [@Linzy19](https://github.com/Linzy19)
- [x] Add accelerate to automatically manage training.
- [x] Joint training with images.
- [ ] Implement [MaskDiT](https://github.com/Anima-Lab/MaskDiT) technique for fast training. 🙏 **[Need your contribution]**
- [ ] Incorporate [NaViT](https://arxiv.org/abs/2307.06304). 🙏 **[Need your contribution]**
- [ ] Add [FreeNoise](https://github.com/arthur-qiu/FreeNoise-LaVie) support for training-free longer video generation. 🙏 **[Need your contribution]**

#### Conduct text2video experiments on landscape dataset.
- [x] Load pretrained weights from [Latte](https://github.com/Vchitect/Latte).
- [ ] Implement [PeRFlow](https://github.com/magic-research/piecewise-rectified-flow) for improving the sampling process. 🙏 **[Need your contribution]**
- [x] Finish data loading, pre-processing utils.
- [x] Add T5 support. 
- [x] Add CLIP support. 🤝 Thanks to [@Ytimed2020](https://github.com/Ytimed2020)
- [x] Add text2image training script.
- [ ] Add prompt captioner. 
  - [ ] Collect training data.
    - [ ] Need video-text pairs with caption. 🙏 **[Need your contribution]**
    - [ ] Extract multi-frame descriptions by large image-language models. 🤝 Thanks to [@HowardLi1984](https://github.com/HowardLi1984)
    - [ ] Extract video description by large video-language models. 🙏 **[Need your contribution]**
    - [ ] Integrate captions to get a dense caption by using a large language model, such as GPT-4. 🤝 Thanks to [@HowardLi1984](https://github.com/HowardLi1984)
  - [ ] Train a captioner to refine captions. 🚀 **[Require more computation]**

#### Train the 1080p model on video2text dataset
- [ ] Looking for a suitable dataset, welcome to discuss and recommend. 🙏 **[Need your contribution]**
- [ ] Add synthetic video created by game engines or 3D representations. 🙏 **[Need your contribution]**
- [x] Finish data loading, and pre-processing utils.
- [x] Support memory friendly training.
  - [x] Add flash-attention2 from pytorch.
  - [x] Add xformers.  🤝 Thanks to [@jialin-zhao](https://github.com/jialin-zhao)
  - [x] Support mixed precision training.
  - [x] Add gradient checkpoint.
  - [x] Support for ReBased and Ring attention. 🤝 Thanks to [@kabachuha](https://github.com/kabachuha)
  - [x] Train using the deepspeed engine. 🤝 Thanks to [@sennnnn](https://github.com/sennnnn)
- [ ] Train with a text condition. Here we could conduct different experiments: 🚀 **[Require more computation]**
  - [x] Train with T5 conditioning.
  - [ ] Train with CLIP conditioning.
  - [ ] Train with CLIP + T5 conditioning (probably costly during training and experiments).
- [ ] Support Chinese. ⌛ [WIP]

#### Control model with more condition
- [ ] Incorporating [ControlNet](https://github.com/lllyasviel/ControlNet). ⌛ [WIP] 🙏 **[Need your contribution]**
- [ ] Incorporating [ReVideo](https://github.com/MC-E/ReVideo). ⌛ [WIP]

</details>
 
## 📂 Repo structure (WIP)
```
├── README.md
├── docs
│   ├── Data.md                    -> Datasets description.
│   ├── Contribution_Guidelines.md -> Contribution guidelines description.
├── scripts                        -> All scripts.
├── opensora
│   ├── dataset
│   ├── models
│   │   ├── ae                     -> Compress videos to latents
│   │   │   ├── imagebase
│   │   │   │   ├── vae
│   │   │   │   └── vqvae
│   │   │   └── videobase
│   │   │       ├── vae
│   │   │       └── vqvae
│   │   ├── captioner
│   │   ├── diffusion              -> Denoise latents
│   │   │   ├── diffusion         
│   │   │   ├── dit
│   │   │   ├── latte
│   │   │   └── unet
│   │   ├── frame_interpolation
│   │   ├── super_resolution
│   │   └── text_encoder
│   ├── sample
│   ├── train                      -> Training code
│   └── utils
```

## 🛠️ Requirements and Installation

1. Clone this repository and navigate to Open-Sora-Plan folder
```
git clone https://github.com/PKU-YuanGroup/Open-Sora-Plan
cd Open-Sora-Plan
```
2. Install required packages
```
conda create -n opensora python=3.8 -y
conda activate opensora
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
4. Install optional requirements such as static type checking:
```
pip install -e '.[dev]'
```

## 🗝️ Usage


### 🤗 Demo

#### Gradio Web UI  <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> 

Highly recommend trying out our web demo by the following command. We also provide [online demo](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.1.0) [![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.1.0). 

<details>
<summary>v1.0.0</summary>
  
Highly recommend trying out our web demo by the following command. We also provide [online demo](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0) [![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0) and [![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/fffiloni/Open-Sora-Plan-v1-0-0) in Huggingface Spaces. 

🤝 Enjoying the [![Replicate demo and cloud API](https://replicate.com/camenduru/open-sora-plan-512x512/badge)](https://replicate.com/camenduru/open-sora-plan-512x512) and [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/Open-Sora-Plan-jupyter/blob/main/Open_Sora_Plan_jupyter.ipynb), created by [@camenduru](https://github.com/camenduru), who generously supports our research!

</details>

```bash
python -m opensora.serve.gradio_web_server
```

#### CLI Inference

```bash
sh scripts/text_condition/sample_video.sh
```

### Datasets
Refer to [Data.md](docs/Data.md)

###  Evaluation
Refer to the document [EVAL.md](docs/EVAL.md).

### CausalVideoVAE

#### Reconstructing

Example:

```Python
python examples/rec_imvi_vae.py --video_path test_video.mp4 --rec_path output_video.mp4 --fps 24 --resolution 512 --crop_size 512 --num_frames 128 --sample_rate 1 --ae CausalVAEModel_4x8x8 --model_path pretrained_488_release --enable_tiling --enable_time_chunk
```

Parameter explanation:

- `--enable_tiling`: This parameter is a flag to enable a tiling conv.

#### Training and Eval

Please refer to the document [CausalVideoVAE](docs/Train_And_Eval_CausalVideoVAE.md).

### VideoGPT VQVAE

Please refer to the document [VQVAE](docs/VQVAE.md).

### Video Diffusion Transformer

#### Training
```
sh scripts/text_condition/train_videoae_65x512x512.sh
```
```
sh scripts/text_condition/train_videoae_221x512x512.sh
```
```
sh scripts/text_condition/train_videoae_513x512x512.sh
```

<!--
## 🚀 Improved Training Performance

In comparison to the original implementation, we implement a selection of training speed acceleration and memory saving features including gradient checkpointing, mixed precision training, and pre-extracted features, xformers, deepspeed. Some data points using **a batch size of 1 with a A100**:
 
### 64×32×32 (origin size: 256×256×256)

| gradient checkpointing | mixed precision | xformers | feature pre-extraction | deepspeed config | compress kv | training speed | memory       |
|:----------------------:|:---------------:|:--------:|:----------------------:|:----------------:|:--------------:|:------------:|:------------:|
| ✔                     | ✔               | ✔        | ✔                     | ❌               | ❌            |0.64 steps/sec  |   43G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2             | ❌            |0.66 steps/sec  |   14G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2             | ✔             |0.66 steps/sec  |   15G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2 offload     | ❌            |0.33 steps/sec  |   11G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2 offload     | ✔             |0.31 steps/sec  |   12G        |

### 128×64×64 (origin size: 512×512×512)

| gradient checkpointing | mixed precision | xformers | feature pre-extraction | deepspeed config | compress kv | training speed | memory       |
|:----------------------:|:---------------:|:--------:|:----------------------:|:----------------:|:--------------:|:------------:|:------------:|
| ✔                     | ✔               | ✔        | ✔                     | ❌               | ❌            |0.08 steps/sec  |   77G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2             | ❌            |0.08 steps/sec  |   41G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2             | ✔             |0.09 steps/sec  |   36G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2 offload     | ❌            |0.07 steps/sec  |   39G        |
| ✔                     | ✔               | ✔        | ✔                     | Zero2 offload     | ✔             |0.07 steps/sec  |   33G        |

-->

## 💡 How to Contribute to the Open-Sora Plan Community
We greatly appreciate your contributions to the Open-Sora Plan open-source community and helping us make it even better than it is now!

For more details, please refer to the [Contribution Guidelines](docs/Contribution_Guidelines.md)




## 👍 Acknowledgement
* [Latte](https://github.com/Vchitect/Latte): The **main codebase** we built upon and it is an wonderful video generated model.
* [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.
* [ShareGPT4Video](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4Video): Improving Video Understanding and Generation with Better Captions.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT): Video Generation using VQ-VAE and Transformers.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [FiT](https://github.com/whlzy/FiT): Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595): Extending Context Window of Large Language Models via Positional Interpolation.


## 🔒 License
* See [LICENSE](LICENSE) for details.

<!--
## ✨ Star History

[![Star History](https://api.star-history.com/svg?repos=PKU-YuanGroup/Open-Sora-Plan)](https://star-history.com/#PKU-YuanGroup/Open-Sora-Plan&Date)
-->


## ✏️ Citing

### BibTeX

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
### Latest DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10948109.svg)](https://zenodo.org/records/10948109)

## 🤝 Community contributors

<a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/Open-Sora-Plan" />
</a>
