# Open-Sora-Plan

[[Project Page]]()

![The architecture of Open-Sora-Plan](assets/framework.jpg)

This project aim to create a simple and scalable repo, to reproduce [Sora](https://openai.com/sora) (Open AI T2V model) and build knowedge about VideoVAE + DiT at scale. But we only have limited resource, we deeply wish the all open source community can contribute to this project.

## News
**[2024.03.01]** Training code are available now! Welcome to watch 👀 this repository for the latest updates.

## Todo

- [x] support variable aspect ratios, resolutions, durations training
- [x] add class-conditioning on embeddings

- [ ] sampling script
- [ ] fine-tune Video-VQVAE on higher resolution
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

## Usage

### Datasets

Refer to [DATA.md](docs/DATA.md)

### Video-VQVAE (VideoGPT)

#### Training

Use the `scripts/train_vqvae.py` script to train a Video-VQVAE. Execute `python scripts/train_vqvae.py -h` for information on all available training settings. A subset of more relevant settings are listed below, along with default values.
```
cd VideoGPT
```

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
python VideoGPT/rec_video.py --video-path "assets/origin_video_0.mp4" --rec-path "rec_video_0.mp4" --num-frames 500 --sample-rate 1
```
```Python
python VideoGPT/rec_video.py --video-path "assets/origin_video_1.mp4" --rec-path "rec_video_1.mp4" --resolution 196 --num-frames 600 --sample-rate 1
```




https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/c15a5ee3-b2bd-4b6d-8460-8b679e4fe182

https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/ce03c5bc-742e-4dfe-859f-96777f0f06c4








### VideoDiT (DiT)

#### Training
```
cd DiT
torchrun  --nproc_per_node=8 train.py \
  --model DiT-XL/122 --pt-ckpt DiT-XL-2-256x256.pt \
  --vae ucf101_stride4x4x4 \
  --data-path /remote-home/yeyang/UCF-101 --num-classes 101 \
  --sample-rate 2 --num-frames 8 --max-image-size 128 \
  --epochs 1400 --global-batch-size 256 --lr 1e-4 \
  --ckpt-every 1000 --log-every 1000 
```

<p align="center">
<img src="assets/loss.jpg" width=50%>
</p>

#### Sampling
Coming soon.

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
