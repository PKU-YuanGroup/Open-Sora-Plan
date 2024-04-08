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

Taking the release version model configuration file `release.json` as an example:

```json
{
  "_class_name": "CausalVAEModel",
  "_diffusers_version": "0.27.2",
  "attn_resolutions": [],
  "decoder_attention": "AttnBlock3D",
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
    "TimeUpsample2x",
    "TimeUpsample2x"
  ],
  "double_z": true,
  "dropout": 0.0,
  "embed_dim": 4,
  "encoder_attention": "AttnBlock3D",
  "encoder_conv_in": "CausalConv3d",
  "encoder_conv_out": "CausalConv3d",
  "encoder_mid_resnet": "ResnetBlock3D",
  "encoder_resnet_blocks": [
    "ResnetBlock3D",
    "ResnetBlock3D",
    "ResnetBlock3D",
    "ResnetBlock3D"
  ],
  "encoder_spatial_downsample": [
    "SpatialDownsample2x",
    "SpatialDownsample2x",
    "SpatialDownsample2x",
    ""
  ],
  "encoder_temporal_downsample": [
    "TimeDownsample2x",
    "TimeDownsample2x",
    "",
    ""
  ],
  "hidden_size": 128,
  "hidden_size_mult": [
    1,
    2,
    4,
    4
  ],
  "loss_params": {
    "disc_start": 2001,
    "disc_weight": 0.5,
    "kl_weight": 1e-06,
    "logvar_init": 0.0
  },
  "loss_type": "opensora.models.ae.videobase.losses.LPIPSWithDiscriminator",
  "lr": 1e-05,
  "num_res_blocks": 2,
  "q_conv": "CausalConv3d",
  "resolution": 256,
  "z_channels": 4
}
```

It configures the modules used in different layers of the encoder and decoder, as well as the loss. By changing the model configuration file, it is easy to train different model structures.

### Training Script

For specific parameters, see the code in `opensora/train/train_causalvae.py`

# Evaluation

In the repository, we can use the script `scripts/causalvae/gen_video.sh` to generate videos and evaluate the generated videos through the script `scripts/causalvae/eval.sh`. Please check the scripts for detailed parameters. Currently, it supports common metrics such as `lpips`, `flolpips`, `ssim`, `psnr`, etc.

# How to Import a Trained Model

Our model class inherits from the configuration and model management classes of huggingface, supporting the download and loading of models from huggingface. It can also import models trained with pytorch lightning.

```
model = CausalVAEModel.from_pretrained(args.ckpt)
model = model.to(device)
```
