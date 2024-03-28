# CausalVideoVAE Report

## Examples

### Image Reconstruction

Resconstruction in **1536×1024**.

<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/1684c3ec-245d-4a60-865c-b8946d788eb9" width="45%"/> <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/46ef714e-3e5b-492c-aec4-3793cb2260b5" width="45%"/>



### Video Reconstruction

We reconstruct two videos with **720×1280**. Since github can't put too big video, we put it here: [1](https://streamable.com/gqojal), [2](https://streamable.com/6nu3j8). 

https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/c100bb02-2420-48a3-9d7b-4608a41f14aa

https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/8aa8f587-d9f1-4e8b-8a82-d3bf9ba91d68

## Model Structure

![image](https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/e3c8b35d-a217-4d96-b2e9-5c248a2859c8)


The Causal Video VAE architecture inherits from the [Stable-Diffusion Image VAE](https://github.com/CompVis/stable-diffusion/tree/main). To ensure that the pretrained weights of the Image VAE can be seamlessly applied to the Video VAE, the model structure has been designed as follows:

**1. CausalConv3D**: Converting Conv2D to CausalConv3D enables joint training of image and video data. CausalConv3D applies a special treatment to the first frame, as it does not have access to subsequent frames. For more specific details, please refer to https://github.com/PKU-YuanGroup/Open-Sora-Plan/pull/145

**2. Initialization**: There are two common [methods](https://github.com/hassony2/inflated_convnets_pytorch/blob/master/src/inflate.py#L5) to expand Conv2D to Conv3D: average initialization and center initialization. But we employ a specific initialization method (tail initialization). This initialization method ensures that without any training, the model is capable of directly reconstructing images, and even videos.

## Training Details

<img width="833" alt="image" src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/88202804/9ffb6dc4-23f6-4274-a066-bbebc7522a14">

We present the loss curves for two distinct initialization methods under 17×256×256. The yellow curve represents the loss using tail init, while the blue curve corresponds to the loss from center initialization. As shown in the graph, tail initialization demonstrates better performance on the loss curve. Additionally, **we found that center initialization leads to error accumulation**, causing the collapse over extended durations.
