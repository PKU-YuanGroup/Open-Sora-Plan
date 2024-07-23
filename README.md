

<h1 align="left"> <a href="">Open-Sora Plan</a></h1>

This project aims to create a simple and scalable repo, to reproduce [Sora](https://openai.com/sora) (OpenAI, but we prefer to call it "ClosedAI" ). We wish the open-source community can contribute to this project. Pull requests are welcome!!!

本项目希望通过开源社区的力量复现Sora，由北大-兔展AIGC联合实验室共同发起，当前版本离目标差距仍然较大，仍需持续完善和快速迭代，欢迎Pull request！！！

<h5 align="left">
  
[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/YtsBNg7n)
[![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/issues/53#issuecomment-1987226516)
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.1.0)
[![Twitter](https://img.shields.io/badge/-Twitter@LinBin46984-black?logo=twitter&logoColor=1D9BF0)](https://x.com/LinBin46984/status/1795018003345510687) <br>
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
</h5>
<h5 align="left"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


# 📣 News
* **[2024.07.25]** 🔥🔥🔥 v1.2.0 is here! Utilizing a 3D full attention architecture instead of 2+1D. We released a true 3D video diffusion model trained on 4s 720p. Checking out our latest [report](docs/Report-v1.2.0.md).
* **[2024.05.27]** 🎉 We are launching Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out our latest [report](docs/Report-v1.1.0.md). Thanks to [ShareGPT4Video's](https://sharegpt4video.github.io/) capability to annotate long videos.
* **[2024.04.09]** 🤝 Excited to share our latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), which learns real-world physics knowledge from time-lapse videos.
* **[2024.04.07]** 🎉🎉🎉 Today, we are thrilled to present Open-Sora-Plan v1.0.0, which significantly enhances video generation quality and text control capabilities. See our [report](docs/Report-v1.0.0.md). Thanks to HUAWEI NPU for supporting us.
* **[2024.03.27]** 🚀🚀🚀 We release the report of [VideoCausalVAE](docs/CausalVideoVAE.md), which supports both images and videos. We present our reconstructed video in this demonstration as follows. The text-to-video model is on the way.
* **[2024.03.01]** 🤗 We launched a plan to reproduce Sora, called Open-Sora Plan! Welcome to **watch** 👀 this repository for the latest updates.

# 😍 Gallery


<table class="center">
<tr>
  <td style="text-align:center;"><b>3D animation of a small, round, fluffy creature with big, expressive eyes explores ...</b></td>
  <td style="text-align:center;"><b>A single drop of liquid metal falls from a floating orb, landing on a mirror-like ...</b></td>
</tr>
<tr>
  <td><video src="" autoplay></td>
  <td><video src="" autoplay></td>
</tr>
<tr>
  <td style="text-align:center;"><b>A drone camera circles around a beautiful historic church built on a rocky outcropping ...</b></td>
  <td style="text-align:center;"><b>Aerial view of Santorini during the blue hour, showcasing the stunning architecture ...</b></td>
</tr>
<tr>
  <td><video src="" autoplay></td>
  <td><video src="" autoplay></td>
</tr>
<tr>
  <td style="text-align:center;"><b>A snowy forest landscape with a dirt road running through it. The road is flanked by ...</b></td>
  <td style="text-align:center;"><b>Drone shot along the Hawaii jungle coastline, sunny day. Kayaks in the water.</b></td>
</tr>
<tr>
  <td><video src="" autoplay></td>
  <td><video src="" autoplay></td>
</tr>
</table>
    
# 😮 Highlights

Open-Sora Plan shows excellent performance in video generation.

### 🔥 High performance CausalVideoVAE, but with fewer training cost
- High compression ratio with excellent performance, capable of **compressing videos by 256 times (4×8×8)**. Causal convolution supports simultaneous inference of images and videos.

### 🚀 Video Diffusion Model based on 3D attention, joint learning of spatiotemporal features.
- With a **3D full attention architecture** instead of a 2+1D model, 3D attention can better capture joint spatial and temporal features.

<p align="center">
    <img src="https://s21.ax1x.com/2024/07/22/pk7cob8.png" width="650" style="margin-bottom: 0.2;"/>
<p>

# 🤗 Demo

### Gradio Web UI

Highly recommend trying out our web demo by the following command. We also provide [online demo](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.1.0). 

```bash
python -m opensora.serve.gradio_web_server --version 221x512x512
```

### ComfyUI

Coming soon...

# 🐳 Model Zoo

| Model | Transformers(HF) |
|---|---|
|||

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
3. Install additional packages for training cases
```
pip install -e ".[train]"
```
4. Install optional requirements such as static type checking:
```
pip install -e '.[dev]'
```

# 🗝️ Training & Validating

## 🗜️ CausalVideoVAE

### Data prepare
The organization of the training data is easy. We only need to put all the videos recursively in a directory. This makes the training more convenient when using multiple datasets.
``` shell
Training Dataset
|——sub_dataset1
    |——sub_sub_dataset1
        |——video1.mp4
        |——video2.mp4
        ......
    |——sub_sub_dataset2
        |——video3.mp4
        |——video4.mp4
        ......
|——sub_dataset2
    |——video5.mp4
    |——video6.mp4
    ......
|——video7.mp4
|——video8.mp4
```
### Training
``` shell
bash scripts/train_vae.sh
```
We introduce the important args for training.
| Argparse | Usage |
|:---|:---|
|_Training size_||
|`--num_frames`|The number of using frames for training videos|
|`--resolution`|The resolution of the input to the VAE|
|`--batch_size`|The local batch size in each GPU|
|`--sample_rate`|The frame interval of when loading training videos|
|_Data processing_||
|`--video_path`|/path/to/dataset|
|_Load weights_||
|`--model_config`|/path/to/config.json The model config of VAE. If you want to train from scratch use this parameter.|
|`--pretrained_model_name_or_path`|A directory containing a model checkpoint and its config. Using this parameter will only load its weight but not load the state of the optimizer|
|`--resume_from_checkpoint`|/path/to/checkpoint It will resume the training process from the checkpoint including the weight and the optimizer.|
### Inference
``` shell
bash scripts/rec_causalvideo_vae.sh
```
We introduce the important args for inference.
| Argparse | Usage |
|:---|:---|
|_Ouoput video size_||
|`--num_frames`|The number of frames of generated videos|
|`--resolution`|The resolution of generated videos|
|_Data processing_||
|`--real_video_dir`|The directory of the original videos.|
|`--generated_video_dir`|The directory of the generated videos.|
|_Load weights_||
|`--ckpt`|/path/to/model_dir. A directory containing the checkpoint of VAE is used for inference and its model config.|
|_Other_||
|`--enable_tilintg`|Use tiling to deal with videos of high resolution and long time.|
|`--output_origin`|Output the original video clips, fed into the VAE.|

For evaluation, you should save the original video clips by using `--output_origin`.
### Evaluation
We introduce the important args in the script for evaluation.
``` shell
bash scripts/eval.sh
```
| Argparse | Usage |
|:---|:---|
|`--metric`|The metric, such as psnr, ssim, lpips|
|`--real_video_dir`|The directory of the original videos.|
|`--generated_video_dir`|The directory of the generated videos.|

## 📜 Text-to-Video 

### Data prepare
We use a `data.txt` file to specify all the training data. Each line in the file consists of `DATA_ROOT` and `DATA_JSON`. The example of `data.txt` is as follows.
```
/path/to/data_root_1,/path/to/data_json_1.json
/path/to/data_root_2,/path/to/data_json_2.json
...
```
Then, we introduce the format of the annotation json file. The absolute data path is the concatenation of `DATA_ROOT` and the `"path"` field in the annotation json file.
#### For image
The format of image annotation file is as follows.
```
[
  {
    "path": "00168/001680102.jpg",
    "cap": [
      "xxxxx."
    ],
    "resolution": {
      "height": 512,
      "width": 683
    }
  },
  ...
]
```

#### For video
The format of image annotation file is as follows.
```
[
  {
    "path": "panda70m_part_5565/qLqjjDhhD5Q/qLqjjDhhD5Q_segment_0.mp4",
    "cap": [
      "A man and a woman are sitting down on a news anchor talking to each other."
    ],
    "resolution": {
      "height": 720,
      "width": 1280
    },
    "fps": 29.97002997002997,
    "duration": 11.444767
  },
  ...
]
```

### Training
```
bash scripts/text_condition/gpu/train_t2v.sh
```

We introduce some key parameters in order to customize your training process.

| Argparse | Usage |
|:---|:---|
|_Training size_||
|`--num_frames 61`|To train videos of different durations, e.g, 29, 61, 93, 125...|
|`--max_height 640`|To train videos of different resolutions|
|`--max_width 480`|To train videos of different resolutions|
|_Data processing_||
|`--data /path/to/data.txt`|Specify your training data.|
|`--speed_factor 1.25`|To accelerate 1.25x videos. |
|`--drop_short_ratio 1.0`|Do not want to train on videos of dynamic durations, discard all video data with frame counts not equal to `--num_frames`|
|`--group_frame`|If you want to train with videos of dynamic durations, we highly recommend specifying `--group_frame` as well. It improves computational efficiency during training.|
|_Multi-stage transfer learning_||
|`--interpolation_scale_h 1.0`|When training a base model, such as 240p (`--max_height 240`, `--interpolation_scale_h 1.0`) , and you want to initialize higher resolution models like 480p (height 480) from 240p's weights, you need to adjust `--max_height 480`, `--interpolation_scale_h 2.0`, and set `--pretrained` to your 240p weights path (path/to/240p/xxx.safetensors).|
|`--interpolation_scale_w 1.0`|Same as `--interpolation_scale_h 1.0`|
|_Load weights_||
|`--pretrained`|This is typically used for loading pretrained weights across stages, such as using 240p weights to initialize 480p training. Or when switching datasets and you do not want the previous optimizer state.|
|`--resume_from_checkpoint`|It will resume the training process from the latest checkpoint in `--output_dir`. Typically, we set `--resume_from_checkpoint="latest"`, which is useful in cases of unexpected interruptions during training.|
|_Sequence Parallelism_||
|`--sp_size 8 --train_sp_batch_size 2`|It means running a batch size of 2 across 8 GPUs (8 GPUs on the same node).|

> [!Warning]
> <div align="left">
> <b>
> 🚨 We have two ways to load weights: `--pretrained` and `--resume_from_checkpoint`. The latter will override the former.
> </b>
> </div>

### Inference

We provide multiple inference scripts to accommodate various requirements.

#### 🖥️ 1 GPU 
If you only have one GPU, it will perform inference on each sample sequentially, one at a time.
```
bash scripts/text_condition/gpu/sample_t2v.sh
```

#### 🖥️🖥️ Multi-GPUs 
If you want to batch infer a large number of samples, each GPU will infer one sample.
```
bash scripts/text_condition/gpu/sample_t2v_ddp.sh
```

#### 🖥️🖥️ Multi-GPUs & Sequence Parallelism 
If you want to quickly infer one sample, it will utilize all GPUs simultaneously to infer that sample.
```
bash scripts/text_condition/gpu/sample_t2v_sp.sh
```

## 🖼️ Image-to-Video

### Data prepare

### Training

### Inference

# 💡 How to Contribute to the Open-Sora Plan Community
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

<!--
## ✨ Star History

[![Star History](https://api.star-history.com/svg?repos=PKU-YuanGroup/Open-Sora-Plan)](https://star-history.com/#PKU-YuanGroup/Open-Sora-Plan&Date)
-->


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

