# VQVAE Documentation

# Introduction

Vector Quantized Variational AutoEncoders (VQ-VAE) is a type of autoencoder that uses a discrete latent representation. It is particularly useful for tasks that require discrete latent variables, such as text-to-speech and video generation.

# Usage

## Initialization

To initialize a VQVAE model, you can use the `VideoGPTVQVAE` class. This class is a part of the `opensora.models.ae` module.

```python
from opensora.models.ae import VideoGPTVQVAE

vqvae = VideoGPTVQVAE()
```

### Training

To train the VQVAE model, you can use the `train_videogpt.sh` script. This script will train the model using the parameters specified in the script.

```bash
bash scripts/videogpt/train_videogpt.sh
```

### Loading Pretrained Models

You can load a pretrained model using the `download_and_load_model` method. This method will download the checkpoint file and load the model.

```python
vqvae = VideoGPTVQVAE.download_and_load_model("bair_stride4x2x2")
```

Alternatively, you can load a model from a checkpoint using the `load_from_checkpoint` method.

```python
vqvae = VQVAEModel.load_from_checkpoint("results/VQVAE/checkpoint-1000")
```

### Encoding and Decoding

You can encode a video using the `encode` method. This method will return the encodings and embeddings of the video.

```python
encodings, embeddings = vqvae.encode(x_vae, include_embeddings=True)
```

You can reconstruct a video from its encodings using the decode method.

```python
video_recon = vqvae.decode(encodings)
```

## Testing

You can test the VQVAE model by reconstructing a video. The `examples/rec_video.py` script provides an example of how to do this.