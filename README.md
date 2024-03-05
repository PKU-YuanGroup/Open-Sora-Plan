# Open-Sora Plan

[[Project Page]](https://pku-yuangroup.github.io/Open-Sora-Plan/) [[中文主页]](https://pku-yuangroup.github.io/Open-Sora-Plan/blog_cn.html)

## Goal
This project aims to create a simple and scalable repo, to reproduce [Sora](https://openai.com/sora) (OpenAI, but we prefer to call it "CloseAI" ) and build knowledge about Video-VQVAE (VideoGPT) + DiT at scale. However, we have limited resources, we deeply wish all open-source community can contribute to this project. Pull request are welcome!!!

本项目希望通过开源社区的力量复现Sora，由北大-兔展AIGC联合实验室共同发起，当前我们资源有限仅搭建了基础架构，无法进行完整训练，希望通过开源社区逐步增加模块并筹集资源进行训练，当前版本离目标差距巨大，仍需持续完善和快速迭代，欢迎Pull request！！！

Project stages:
- Primary
1. Setup the codebase and train a un-conditional model on landscape dataset.
2. Train models that boost resolution and duration.

- Extensions
3. Conduct text2video experiments on landscape dataset.
4. Train the 1080p model on video2text dataset.
5. Control model with more condition.


<div style="display: flex; justify-content: center;"> 
  <img src="assets/we_want_you.jpg" width=200> 
  <img src="assets/framework.jpg" width=600> 
</div>

  
## News
**[2024.03.04]** We re-organize and modulize our codes and make it easy to [contribute](https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file#how-to-contribute-to-the-open-sora-plan-community) to the project, please see the [Repo structure](https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file#repo-structure).

**[2024.03.03]** We open some [discussions](https://github.com/PKU-YuanGroup/Open-Sora-Plan/discussions) and clarify several issues.

**[2024.03.01]** Training codes are available now! Learn more in our [project page](https://pku-yuangroup.github.io/Open-Sora-Plan/). Please feel free to watch 👀 this repository for the latest updates.


## Todo

- [x] support variable aspect ratios, resolutions, durations training on DiT
- [x] dynamic mask input
- [x] add class-conditioning on embeddings

- [ ] sampling script
- [ ] add positional interpolation
- [ ] fine-tune Video-VQVAE on higher resolution
- [ ] incorporating SiT
- [ ] incorporating more conditions
- [ ] training with more data and more GPU

## Repo structure
```
├── README.md
├── docs
│   ├── Data.md                    -> Datasets description.
│   ├── Contribution_Guidelines.md -> Contribution guidelines description.
├── scripts                        -> All training scripts.
│   └── train.sh
├── sora
│   ├── dataset                    -> Dataset code to read videos
│   ├── models 
│   │   ├── captioner               
│   │   ├── super_resolution        
│   ├── modules
│   │   ├── ae                     -> compress videos to latents
│   │   │   ├── vqvae
│   │   │   ├── vae
│   │   ├── diffusion              -> denoise latents
│   │   │   ├── dit
│   │   │   ├── unet
|   ├── utils.py                   
│   ├── train.py                   -> Training code
```

## Requirements and Installation

The recommended requirements are as follows.

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

## Usage

### Datasets
Refer to [Data.md](docs/Data.md)


### Video-VQVAE (VideoGPT)

#### Training

```
cd src/sora/modules/ae/vqvae/videogpt
```

Refer to origin [repo](https://github.com/wilson1yan/VideoGPT?tab=readme-ov-file#training-vq-vae). Use the `scripts/train_vqvae.py` script to train a Video-VQVAE. Execute `python scripts/train_vqvae.py -h` for information on all available training settings. A subset of more relevant settings are listed below, along with default values.

##### VQ-VAE Specific Settings
* `--embedding_dim`: number of dimensions for codebooks embeddings
* `--n_codes 2048`: number of codes in the codebook
* `--n_hiddens 240`: number of hidden features in the residual blocks
* `--n_res_layers 4`: number of residual blocks
* `--downsample 4 4 4`: T H W downsampling stride of the encoder

##### Training Settings
* `--gpus 2`: number of gpus for distributed training
* `--sync_batchnorm`: uses `SyncBatchNorm` instead of `BatchNorm3d` when using > 1 gpu
* `--gradient_clip_val 1`: gradient clipping threshold for training
* `--batch_size 16`: batch size per gpu
* `--num_workers 8`: number of workers for each DataLoader

##### Dataset Settings
* `--data_path <path>`: path to an `hdf5` file or a folder containing `train` and `test` folders with subdirectories of videos
* `--resolution 128`: spatial resolution to train on 
* `--sequence_length 16`: temporal resolution, or video clip length

#### Reconstructing

```Python
python rec_video.py --video-path "assets/origin_video_0.mp4" --rec-path "rec_video_0.mp4" --num-frames 500 --sample-rate 1
```
```Python
python rec_video.py --video-path "assets/origin_video_1.mp4" --rec-path "rec_video_1.mp4" --resolution 196 --num-frames 600 --sample-rate 1
```


We present four reconstructed videos in this demonstration, arranged from left to right as follows: 


| **3s 596x336** | **10s 256x256** | **18s 196x196**  | **24s 168x96** |
| --- | --- | --- | --- |
| <img src="assets/rec_video_2.gif">  | <img src="assets/rec_video_0.gif">  | <img src="assets/rec_video_1.gif">  | <img src="assets/rec_video_3.gif"> |

### VideoDiT (DiT)

#### Training
```
sh scripts/train.sh
```

<p align="center">
<img src="assets/loss.jpg" width=60%>
</p>

#### Sampling
Coming soon.

## How to Contribute to the Open-Sora Plan Community
We greatly appreciate your contributions to the Open-Sora Plan open-source community and helping us make it even better than it is now!

For more details, please refer to the [Contribution Guidelines](docs/Contribution_Guidelines.md)

## Acknowledgement
* [DiT](https://github.com/facebookresearch/DiT/tree/main): Scalable Diffusion Models with Transformers.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT): Video Generation using VQ-VAE and Transformers.
* [FiT](https://github.com/whlzy/FiT): Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595): Extending Context Window of Large Language Models via Positional Interpolation.

## License
* The service is a research preview intended for non-commercial use only. See [LICENSE.txt](LICENSE.txt) for details.

## Contributors

<a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/Open-Sora-Plan" />
</a>
