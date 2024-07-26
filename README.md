

<h1 align="left"> <a href="">Open-Sora Plan</a></h1>

This project aims to create a simple and scalable repo, to reproduce [Sora](https://openai.com/sora) (OpenAI, but we prefer to call it "ClosedAI" ). We wish the open-source community can contribute to this project. Pull requests are welcome! The current code supports complete training and inference using the Huawei Ascend AI computing system. Models trained on Huawei Ascend can also output video quality comparable to industry standards.

本项目希望通过开源社区的力量复现Sora，由北大-兔展AIGC联合实验室共同发起，当前版本离目标差距仍然较大，仍需持续完善和快速迭代，欢迎Pull request！目前代码同时支持使用国产AI计算系统（华为昇腾）进行完整的训练和推理。基于昇腾训练出的模型，也可输出持平业界的视频质量。

<h5 align="left">
  
[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/FkFm5M2J)
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
* **[2024.07.24]** 🔥🔥🔥 v1.2.0 is here! Utilizing a 3D full attention architecture instead of 2+1D. We released a true 3D video diffusion model trained on 4s 720p. Checking out our latest [report](docs/Report-v1.2.0.md).
* **[2024.05.27]** 🎉 We are launching Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out our latest [report](docs/Report-v1.1.0.md). Thanks to [ShareGPT4Video's](https://sharegpt4video.github.io/) capability to annotate long videos.
* **[2024.04.09]** 🤝 Excited to share our latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), which learns real-world physics knowledge from time-lapse videos.
* **[2024.04.07]** 🎉🎉🎉 Today, we are thrilled to present Open-Sora-Plan v1.0.0, which significantly enhances video generation quality and text control capabilities. See our [report](docs/Report-v1.0.0.md). Thanks to HUAWEI NPU for supporting us.
* **[2024.03.27]** 🚀🚀🚀 We release the report of [VideoCausalVAE](docs/CausalVideoVAE.md), which supports both images and videos. We present our reconstructed video in this demonstration as follows. The text-to-video model is on the way.
* **[2024.03.01]** 🤗 We launched a plan to reproduce Sora, called Open-Sora Plan! Welcome to **watch** 👀 this repository for the latest updates.

# 😍 Gallery

93×1280×720 Text-to-Video Generation. The video quality has been compressed for playback on GitHub.

<table class="center">
<tr>
  <td><video src="https://github.com/user-attachments/assets/1c84bc92-d585-46c9-ae7c-e5f79cefea88" autoplay></td>
</tr>
</table>

  
# 😮 Highlights

Open-Sora Plan shows excellent performance in video generation.

### 🔥 High performance CausalVideoVAE, but with fewer training cost
- High compression ratio with excellent performance, capable of **compressing videos by 256 times (4×8×8)**. Causal convolution supports simultaneous inference of images and videos but only need **1 node to train**.

### 🚀 Video Diffusion Model based on 3D attention, joint learning of spatiotemporal features.
- With a **3D full attention architecture** instead of a 2+1D model, 3D attention can better capture joint spatial and temporal features.

<p align="center">
    <img src="https://s21.ax1x.com/2024/07/22/pk7cob8.png" width="650" style="margin-bottom: 0.2;"/>
<p>

# 🤗 Demo

### Gradio Web UI

Highly recommend trying out our web demo by the following command.

```bash
python -m opensora.serve.gradio_web_server --model_path "path/to/model" --ae_path "path/to/causalvideovae"
```

### ComfyUI

Coming soon...

# 🐳 Resource

| Version | Architecture |  Diffusion Model | CausalVideoVAE | Data|
|:---|:---|:---|:---|:---|
| v1.2.0 | 3D | [93x720p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x720p), [29x720p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x720p)[1], [93x480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x480p)[1,2] | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae)| [Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) |
| v1.1.0 | 2+1D | [221x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/221x512x512), [65x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/65x512x512) |[Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/vae) |[Data and Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0)|
| v1.0.0 | 2+1D | [65x512x512](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/65x512x512), [65x256x256](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/65x256x256), [17x256x256](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/17x256x256) | [Anysize](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/vae) | [Data and Annotations](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.0.0)|

> [1] Please note that the weights for v1.2.0 29×720p and 93×480p were trained on Panda70M and have not undergone final high-quality data fine-tuning, so they may produce watermarks.

> [2] We fine-tuned 3.5k steps from 93×720p to get 93×480p for community research use.

> [!Warning]
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
bash scripts/causalvae/train.sh
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
bash scripts/causalvae/rec_video.sh
```
We introduce the important args for inference.
| Argparse | Usage |
|:---|:---|
|_Ouoput video size_||
|`--num_frames`|The number of frames of generated videos|
|`--height`|The resolution of generated videos|
|`--width`|The resolution of generated videos|
|_Data processing_||
|`--video_path`|The path to the original video|
|`--rec_path`|The path to the generated video|
|_Load weights_||
|`--ae_path`|/path/to/model_dir. A directory containing the checkpoint of VAE is used for inference and its model config.json|
|_Other_||
|`--enable_tilintg`|Use tiling to deal with videos of high resolution and long duration|
|`--save_memory`|Save memory to inference but lightly influence quality|


### Evaluation


For evaluation, you should save the original video clips by using `--output_origin`.
``` shell
bash scripts/causalvae/prepare_eval.sh
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


Then, we begin to eval. We introduce the important args in the script for evaluation.
``` shell
bash scripts/causalvae/eval.sh
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
The format of video annotation file is as follows. More details refer to [HF dataset](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0).
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

We provide multiple inference scripts to support various requirements. We recommend configuration `--guidance_scale 7.5 --num_sampling_steps 100 --sample_method EulerAncestralDiscrete` for sampling.

**Inference on 93×720p**, we report speed on H100.

| Size | 1 GPU | 8 GPUs (sp) | 
|---|---|---|
|29×720p|420s/100step|80s/100step|
|93×720p|3400s/100step|450s/100step|

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
Coming soon...

### Training
Coming soon...

### Inference
Coming soon...

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

