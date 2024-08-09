# Training

To execute in the terminal: `bash scripts/causalvae/train.sh`

> When using GAN loss for training, two backward propagations are required. However, when [custom optimizers](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#use-multiple-optimizers-like-gans) are implemented in PyTorch Lightning, it can lead to the training step count being doubled, meaning each training loop effectively results in two steps. This issue can make it counterintuitive when setting the training step count and the starting step count for the GAN loss.

## Code Structure

CausalVideoVAE is located in the directory `opensora/models/ae/videobase`. The directory structure is as follows:

```
.
├── causal_vae
├── causal_vqvae
├── configuration_videobase.py
├── dataset_videobase.py
├── __init__.py
├── losses
├── modeling_videobase.py
├── modules
├── __pycache__
├── trainer_videobase.py
├── utils
└── vqvae
```

The `casual_vae` directory defines the overall structure of the CausalVideoVAE model, and the `modules` directory contains some of the required modules for the model, including **CausalConv3D**, **ResnetBlock3D**, **Attention**, etc. The `losses` directory includes **GAN loss**, **Perception loss**, and other content.

## Configuration

Model training requires two key files: one is the `config.json` file, which configures the model structure, loss function, learning rate, etc. The other is the `train.sh` file, which configures the dataset, training steps, precision, etc.

### Model Configuration File

Taking the v1.1.0 version model configuration file as an example:

```json
{
  "_class_name": "CausalVAEModel",
  "_diffusers_version": "0.27.2",
  "_name_or_path": "../results/pretrained_488_tail",
  "attn_resolutions": [],
  "decoder_attention": "AttnBlock3DFix",
  "decoder_conv_in": "CausalConv3d",
  "decoder_conv_out": "CausalConv3d",
  "decoder_mid_resnet": "ResnetBlock3D",
  "decoder_resnet_blocks": [
    "ResnetBlock3D",
    "ResnetBlock3D",
    "ResnetBlock3D",
    "ResnetBlock3D"
  ],
  "decoder_spatial_upsample": [
    "",
    "SpatialUpsample2x",
    "SpatialUpsample2x",
    "SpatialUpsample2x"
  ],
  "decoder_temporal_upsample": [
    "",
    "",
    "TimeUpsampleRes2x",
    "TimeUpsampleRes2x"
  ],
  "double_z": true,
  "dropout": 0.0,
  "embed_dim": 4,
  "encoder_attention": "AttnBlock3DFix",
  "encoder_conv_in": "Conv2d",
  "encoder_conv_out": "CausalConv3d",
  "encoder_mid_resnet": "ResnetBlock3D",
  "encoder_resnet_blocks": [
    "ResnetBlock2D",
    "ResnetBlock2D",
    "ResnetBlock3D",
    "ResnetBlock3D"
  ],
  "encoder_spatial_downsample": [
    "Downsample",
    "Downsample",
    "Downsample",
    ""
  ],
  "encoder_temporal_downsample": [
    "",
    "TimeDownsampleRes2x",
    "TimeDownsampleRes2x",
    ""
  ],
  "hidden_size": 128,
  "hidden_size_mult": [
    1,
    2,
    4,
    4
  ],
  "in_channels": 3,
  "loss_params": {
    "disc_start": 2001,
    "disc_weight": 0.5,
    "kl_weight": 1e-06,
    "logvar_init": 0.0
  },
  "loss_type": "opensora.models.ae.videobase.losses.LPIPSWithDiscriminator3D",
  "lr": 1e-05,
  "num_res_blocks": 2,
  "out_channels": 3,
  "q_conv": "CausalConv3d",
  "resolution": 256,
  "z_channels": 4
}
```

It configures the modules used in different layers of the encoder and decoder, as well as the loss. By changing the model configuration file, it is easy to train different model structures.

### Training Script

The following is a description of the parameters for the `train_causalvae.py`:

| Parameter                  | Default Value | Description                                            |
|-----------------------------|-----------------|--------------------------------------------------------|
| `--exp_name`                 | "causalvae"      | The name of the experiment, used for the folder where results are saved. |
| `--batch_size`               | 1               | The number of samples per training iteration.                  |
| `--precision`                | "bf16"          | The numerical precision type used for training.                |
| `--max_steps`                | 100000          | The maximum number of steps for the training process.          |
| `--save_steps`               | 2000            | The interval at which to save the model during training.      |
| `--output_dir`               | "results/causalvae" | The directory where training results are saved.          |
| `--video_path`               | "/remote-home1/dataset/data_split_tt" | The path where the video data is stored.            |
| `--video_num_frames`         | 17              | The number of frames per video.                             |
| `--sample_rate`              | 1               | The sampling rate, indicating the number of video frames per second. |
| `--dynamic_sample`            | False           | Whether to use dynamic sampling.                               |
| `--model_config`             | "scripts/causalvae/288.yaml" | The path to the model configuration file.              |
| `--n_nodes`                  | 1               | The number of nodes used for training.                        |
| `--devices`                  | 8               | The number of devices used for training.                       |
| `--resolution`               | 256              | The resolution of the videos.                                 |
| `--num_workers`              | 8               | The number of subprocesses used for data handling.            |
| `--resume_from_checkpoint`   | None            | Resume training from a specified checkpoint.                  |
| `--load_from_checkpoint`     | None            | Load the model from a specified checkpoint.                  |

Please ensure that the values provided for these parameters are appropriate for your training setup.

# Evaluation


1. Video Generation:
The script `scripts/causalvae/gen_video.sh` in the repository is utilized for generating videos. For the parameters, please refer to the script itself.

2. Video Evaluation:
After video generation, You can evaluate the generated videos using the `scripts/causalvae/eval.sh` script. This evaluation script supports common metrics, including lpips, flolpips, ssim, psnr, and more.

> Please note that you must generate the videos before executing the eval script. Additionally, it is essential to ensure that the video parameters used when generating the videos are consistent with those used during the evaluation.

# How to Import a Trained Model

Our model class inherits from the configuration and model management classes of huggingface, supporting the download and loading of models from huggingface. It can also import models trained with pytorch lightning.

```
model = CausalVAEModel.from_pretrained(args.ckpt)
model = model.to(device)
```

