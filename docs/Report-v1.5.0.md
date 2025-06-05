## Report v1.5.0

中文版本Report请参考[Report-v1.5.0_cn.md](Report-v1.5.0_cn.md)

In October 2024, we released Open-Sora Plan v1.3.0, introducing the sparse attention structure, Skiparse Attention, to the field of video generation for the first time. Additionally, we adopted the efficient WFVAE, significantly reducing encoding time and memory usage during training.

In Open-Sora Plan v1.5.0, We introduce several key updates to enhance the framework:

1、Improved Sparse DiT, SUV. Building on Skiparse Attention, we extend sparse DiT into a U-shaped sparse structure. This design preserves speed advantages while enabling sparse DiT to achieve performance comparable to dense DiT.

2、Higher-compression WFVAE. In Open-Sora Plan v1.5.0, we explore a WFVAE with an 8×8×8 downsampling rate. It outperforms the performance of the widely adopted 4×8×8 VAE in the community, while reducing the latent shape by half and shortening the attention sequence length.

3、Data and model scaling. In Open-Sora Plan v1.5.0, we collect 1.1 billion high-quality images and 40 million high-quality videos. The model is scaled up to 8.5 billion parameters, resulting in strong overall performance.

4、Simplified Adaptive Gradient Clipping strategy. Compared to the more complex batch-dropping method in version 1.3.0, version 1.5.0 maintains a simple adaptive gradient norm threshold for clipping, making it more compatible with various parallel training strategies.

Open-Sora Plan v1.5.0 is fully trained and inferred on Ascend 910-series accelerators, using the mindspeed-mm framework to support parallel training strategies.

### Open-Source Release

Open-Sora Plan v1.5.0 is open-sourced with the following components:

1、All training and inference code. You can also find the implementation of Open-Sora Plan v1.5.0 in the official [MindSpeed-MM](https://gitee.com/ascend/MindSpeed-MM) repository.

2、The WFVAE weights with 8×8×8 compression, along with the 8.5B SUV denoiser weights.

## Detailed Technical Report

### Data collection and processing

Our dataset includes 1.1B images from [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B)、[COYO-700M](https://github.com/kakaobrain/coyo-dataset)、[LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/), with no filtering applied aside from resolution checks. The video data are drawn from [Panda-70M](https://github.com/snap-research/Panda-70M) and internal sources, and filtered using the same protocol as in Open-Sora Plan v1.3.0, yielding 40M high-quality videos.

### Adaptive Grad Clipping

In Open-Sora Plan v1.3.0, we introduce an Adaptive Grad Clipping strategy based on discarding gradient-abnormal batches. While highly stable, this method involve overly complex execution logic. In Open-Sora Plan v1.5.0, we optimize the strategy by maintaining the gradient norm threshold via an exponential moving average (EMA). Gradients exceeding the threshold are clipped accordingly. This approach effectively extends the fixed threshold of 1.0, which is commonly used in large-scale models, into a dynamic, training-dependent threshold.

```python
'''
	moving_avg_max_grad_norm: the maximum gradient norm maintained via EMA
	moving_avg_max_grad_norm_var: the variance of the maximum gradient norm maintained via EMA
	clip_threshold: the gradient clipping threshold computed using the 3-sigma rule
	ema_decay: the EMA decay coefficient, typically set to 0.99.
	grad_norm: grad norm at the current step 
'''
clip_threshold = moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)
if grad_norm <= clip_threshold:
    # If the gradient norm is below the clipping threshold, the parameters are updated normally at this step, and both the moving_avg_max_grad_norm and moving_avg_max_grad_norm_var are updated accordingly.
    moving_avg_max_grad_norm = ema_decay * moving_avg_max_grad_norm + (1 - ema_decay) * grad_norm
    max_grad_norm_var = (moving_avg_max_grad_norm - grad_norm) ** 2
    moving_avg_max_grad_norm_var = ema_decay * moving_avg_max_grad_norm_var + (1 - ema_decay) * max_grad_norm_var
    # update weights...
else:
    # If the gradient norm exceeds the clipping threshold, the gradients are first clipped to reduce the norm to the threshold value before updating the parameters.
    clip_coef = grad_norm / clip_threshold
    grads = clip(grads, clip_coef) # clipping grads
    # update weights...
```

Compared to the strategy in v1.3.0, this approach is simpler to implement and effectively addresses the issue of loss spikes that occur in the later stages of diffusion training when the gradient norm is significantly below 1.0.

### WFVAE with 8x8x8 compression

In version 1.5.0, we increase the temporal compression rate of the VAE from 4× to 8×, reducing the latent shape to half that of the previous version. This enables the generation of videos with higher frame counts.

| Model             | THW(C)        | PSNR         | LPIPS         | rFVD         |
| ----------------- | ------------- | ------------ | ------------- | ------------ |
| CogVideoX         | 4x8x8 (16)    | <u>36.38</u> | 0.0243        | <u>50.33</u> |
| StepVideo         | 8x16x16 (16)  | 33.61        | 0.0337        | 113.68       |
| LTXVideo          | 8x32x32 (128) | 33.84        | 0.0380        | 150.87       |
| Wan2.1            | 4x8x8 (16)    | 35.77        | **0.0197**    | **46.05**    |
| Ours （WF-VAE-M） | 8x8x8 (32)    | **36.91**    | <u>0.0205</u> | 52.53        |

**Test on an open-domain dataset with 1K samples.**

For more details on WFVAE, please refer to [WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model](https://arxiv.org/abs/2411.17459)

### Training Text-to-Video Diffusion Model

#### Framework —— SUV: A Sparse U-shaped Diffusion Transformer For Fast Video Generation

In Open-Sora Plan v1.3.0, we discuss the strengths and weaknesses of Full 3D Attention and 2+1D Attention. Based on their characteristics, we propose Skiparse Attention, a novel global sparse attention mechanism. For details, please refer to [Report-v1.3.0](Report-v1.3.0.md).

Under a predefined sparsity $k$, Skiparse Attention selects a subsequence of length $\frac{1}{k}$ of the original sequence in an alternating Single-Skip and Group-Skip pattern for attention interaction. This design approximates the effect of Full 3D Attention. As the sparsity increases, the selected positions become more widely spaced; as it decreases, the positions become more concentrated. Regardless of the sparsity, Skiparse Attention remains global.

In Open-Sora Plan v1.5.0, we interpret this sparse interaction pattern as a form of token-level information downsampling. Sparser Skiparse Attention performs more semantic-level interactions, while denser Skiparse Attention captures fine-grained information. Following the multi-scale design principle in neural networks, we introduce Skiparse Attention with U-shaped sparsity variation: low-sparsity Skiparse Attention is used in shallow layers, with Full 3D Attention applied at the shallowest layer, and high-sparsity Skiparse Attention in deeper layers. Inspired by the UNet architecture, we further incorporate long skip connections between stages with identical sparsity. This U-shaped DiT architecture based on Skiparse Attention is referred to as **SUV**.

![SUV](https://img.picui.cn/free/2025/06/05/684108197cbb8.png)

In Open-Sora Plan v1.5.0, we adopt an SUV architecture based on MMDiT. Skiparse Attention is applied to the video latents, while the text embeddings are only repeated to align with the skiparse-processed latent shape, without any sparsification.

The SUV architecture offers the following advantages:

1、SUV is the first sparsification method proven effective for video generation. Our ablation studies show that it achieves performance comparable to dense DiT within the approximate training steps. Moreover, it can be applied during both pretraining and inference. Testing on the Ascend 910B platform at 121×576×1024 shape shows SUV runs over 35% faster than Dense DiT, with the attention operation alone gaining a speed boost of over 45%.

2、Unlike UNet structures that explicitly downsample feature maps and cause information loss, the U-shaped structure of SUV operates on attention. The shape of the feature map remains unchanged, preserving information while altering only the granularity of token-level interactions.

3、Skiparse Attention and SUV only change the attention computation during the forward pass instead of modifying model weights. This allows dynamic adjustment of sparsity throughout training: lower sparsity for image or low-resolution video training, and higher sparsity for high-resolution video training. As a result, FLOPS grow approximately linearly with increasing of sequence length.


A more detailed analysis of the SUV architecture will be released in a future arXiv update.

#### Training Stage

Our training consists of two stages: Text-to-Image and Text-to-Video.

#### Text-to-Image

Previous studies have shown that image weights trained on synthetic data may negatively impact video training. Therefore, in the v1.5.0 update, we choose to train image weights using a much larger corpus of real-world data, totaling 1.1B images. Since image data come in various resolutions, whereas videos are primarily in a 9:16 aspect ratio, we adopt multi-resolution training for images using five common aspect ratios—(1,1), (3,4), (4,3), (9,16), and (16,9)—along with the Min-Max Token Strategy. In contrast, video training is conducted using a fixed 9:16 resolution.

The difference between Skiparse Attention and Full Attention lies in the token sequences involved in the forward computation; the required weights remain identical. Therefore, we can first train the model using Dense MMDiT with Full 3D Attention, and then fine-tune it to the Sparse MMDiT mode after sufficient training.

**Image-Stage-1:** Training is conducted using 512 Ascend 910B accelerators. We train a randomly initialized Dense MMDiT on 256²-pixel images with multi-resolution enabled. The learning rate is set to 1e-4, with a batch size of 8096. This stage runs for a total of 225k steps.

**Image-Stage-2:** Training is conducted using 384 Ascend 910B accelerators. We train on 384²-pixel images with multi-resolution still enabled. The learning rate remains 1e-4, the batch size is 6144, and training lasts for 150k steps.

**Image-Stage-3:** Training is conducted using 256 Ascend 910B accelerators. We train on 288x512 images with force resolution. The learning rate is 1e-4, the batch size is 4096, and training lasts for 110k steps. This stage completes the Dense MMDiT training.

**Image-Stage-4:** Training is conducted using 256 Ascend 910B accelerators. We initialize the SUV model using the pretrained weights from Dense MMDiT, with skip connections zero-initialized to ensure that the model could produce non-noise outputs at the start. In practice, zero-shot inference reveals that the generated images contained meaningful low-frequency structures. Our experiments confirm that fine-tuning from Dense DiT to SUV converges quickly. This stage uses a fixed resolution of 288×512, a learning rate of 1e-4, a batch size of 4096, and is trained for approximately 160k steps.

#### Text-to-Video

For video training, we fix the aspect ratio at 9:16 and training solely on video data instead of joint training with image data. All training in this stage is performed using 512 Ascend 910B accelerators.

**Video-Stage-1:** Starting from the SUV weights pretrained during the Text-to-Image phase, we train on videos with a shape of 57×288×512 for about 40k steps. The setup includes a learning rate of 6e-5, TP/SP parallelism of 2, gradient accumulation set to 2, a micro batch size of 2, and a global batch size of 1024. Videos are trained at 24 fps, representing approximately 2.4 seconds (57/24 ≈ 2.4s) of content per sample. This stage marks the initial adaptation from image-based to video-based weights, for which shorter video clips are intentionally selected to ensure stable initialization.

**Video-Stage-2:** We further train on videos with a shape of 57×288×512 for 45k steps, keeping the learning rate, TP/SP parallelism, and gradient accumulation settings unchanged. However, the training frame rate is reduced to 12 fps, corresponding to ~4.8 seconds of video content per sample (57/12 ≈ 4.8s). This stage aims to enhance temporal learning without increasing sequence length, serving as preparation for later high-frame-counts training.

**Video-Stage-3:** We train on videos with a shape of 121×288×512 for approximately 25k steps. The learning rate is adjusted to 4e-5, with TP/SP parallelism set to 4, gradient accumulation steps set to 2, a micro batch size of 4, and a global batch size of 1024. In this stage, we revert to a training frame rate of 24 fps.

**Video-Stage-4:** We conduct training on videos with a shape of 121×576×1024 for a total of 16k + 9k steps. The learning rates are set to 2e-5 and 1e-5 for the two phases, respectively. TP/SP parallelism is configured as 4, with gradient accumulation steps set to 4, a micro batch size of 1, and a global batch size of 512.

**Video-Stage-5:** We train on a high-quality subset of the dataset for 5k steps, using a learning rate of 1e-5. TP/SP parallelism is set to 4, with gradient accumulation steps of 4, a micro batch size of 1, and a global batch size of 512.

 #### Performance on Vbench

| Model                      | Total Score   | Quality Score | Semantic Score | **aesthetic quality** |
| -------------------------- | ------------- | ------------- | -------------- | --------------------- |
| Mochi-1                    | 80.13%        | 82.64%        | 70.08%         | 56.94%                |
| CogvideoX-2B               | 80.91%        | 82.18%        | 75.83%         | 60.82%                |
| CogvideoX-5B               | 81.61%        | 82.75%        | 77.04%         | 61.98%                |
| Step-Video-T2V             | 81.83%        | <u>84.46%</u> | 71.28%         | 61.23%                |
| CogvideoX1.5-5B            | 82.17%        | 82.78%        | **79.76%**     | 62.79%                |
| Gen-3                      | 82.32%        | 84.11%        | 75.17%         | <u>63.34%</u>         |
| HunyuanVideo (Open-Source) | **83.24%**    | **85.09%**    | 75.82%         | 60.36%                |
| Open-Sora Plan v1.5.0      | <u>82.95%</u> | 84.15%        | <u>78.17%</u>  | **66.93%**            |

### Training Image-to-Video Diffusion Model

Coming Soon...

### Future Work

Currently, open-source models such as Wan2.1 have achieved performance comparable to closed-source commercial counterparts. Given the gap in computing resources and data availability compared to industry-scale efforts, the future development of the Open-Sora Plan will focus on the following directions:

1、Latents Cache。

In the training process of Text-to-Video models, the data must be processed through two key modules—the Variational Autoencoder (VAE) and the Text Encoder—to extract features from both video/images and their corresponding prompts. These encoded features serve as inputs to the training model. However, in existing industry practices, feature encoding is redundantly performed on the multimodal training dataset during every training epoch. This leads to additional computational overhead and significantly prolongs the total training time.

Specifically, in conventional training pipelines, the VAE and Text Encoder modules are typically kept resident in GPU memory to perform feature encoding in real time during each epoch. While this ensures on-the-fly encoding, it also results in persistently high GPU memory usage, becoming a major bottleneck for training efficiency. This issue is exacerbated when handling large-scale datasets or complex models, where memory constraints further limit model capacity and training speed.

To address the above issue, we propose an optimization strategy that replaces repeated feature computation with feature lookup. The core idea is to decouple feature encoding from model training. Specifically, during pretraining or the first training epoch, we compute and store the most computationally expensive text prompt features in external high-performance storage. During subsequent training, the model directly loads these precomputed features from storage, avoiding redundant encoding operations. This design significantly reduces computational overhead and GPU memory usage, allowing more memory to be allocated to model training.

Based on the following configuration environment, we compare the training time per epoch and per step before and after applying the feature caching strategy. Experimental results show that storing precomputed features reduces multi-epoch training time by approximately 30% and frees up around 20% of GPU memory resources.

| **Configuration** |                 **Details**                 |
| :---------------: | :-----------------------------------------: |
|       Model       | Open-Sora Plan v1.5.0 (2B-level parameters) |
|      Dataset      |         100K images and 10K videos          |
|   Accelerators    |             8× Nvidia A800 GPUs             |
|  Feature Storage  |         Huawei OceanStor AI Storage         |

Test cases:

| **Training Stage** | **Test Type**          | **Batch Size** | **Time per Step** | **Time per Epoch** | **Memory Usage** |
| ------------------ | ---------------------- | -------------- | ----------------- | ------------------ | ---------------- |
| Low-Res Images     | General Method         | 64             | 6.53s             | 21 min 12s         | 56 GB            |
|                    | Feature Caching Method | 64             | 4.10s             | 13 min 19s         | 40 GB            |
|                    | General Method         | 128            | 12.78s            | 20 min 39s         | 74 GB            |
|                    | Feature Caching Method | 128            | 7.81s             | 12 min 38s         | 50 GB            |
| Low-Res Videos     | General Method         | 8              | 8.90s             | 26 min 23s         | 68 GB            |
|                    | Feature Caching Method | 8              | 7.78s             | 23 min 05s         | 51 GB            |
| High-Res Videos    | General Method         | 4              | 17.00s            | 101 min            | 71 GB            |
|                    | Feature Caching Method | 4              | 16.00s            | 97 min             | 57 GB            |

2、Improved DiT pretraining with sparse or linear attention. In v1.3.0, we introduce the first DiT pretrained with sparse attention in the community. This is extended in v1.5.0 into the SUV architecture, enabling sparse DiT to achieve performance comparable to its dense counterpart. While sparse and linear attention have demonstrated significant success in the LLM domain, their application in video generation remains underexplored. In future versions, we plan to further investigate the integration of sparse and linear attention into video generation models.

3、MoE-based DiT. Since the release of [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), the MoE (Mixture-of-Experts) paradigm has become a common approach for scaling LLMs to larger parameter sizes. Currently, open-source video generation models are capped at around 14B parameters, which is still relatively small compared to the 100B+ scales in the LLM field. Incorporating MoE into the DiT architecture, and exploring its combination with sparse and linear attention, is a future direction under consideration by the Open-Sora Plan team.

4、Unified video generation models for both generation and understanding. The March release of GPT-4o demonstrates that unified architectures combining generation and understanding can offer fundamentally different capabilities compared to purely generative models. In the video domain, we should similarly anticipate the potential breakthroughs that such unified generative models might bring.

5、Enhancing Image-to-Video generation models. Current approaches in this field still largely follow either the SVD paradigm or the inpainting-based paradigm adopted since Open-Sora Plan v1.2.0. Both approaches require extensive fine-tuning of pretrained Text-to-Video models. From a practical standpoint, Text-to-Video is more aligned with academic exploration, while Image-to-Video is more relevant to real-world production scenarios. As a result, developing a new paradigm for Image-to-Video will be a key focus for the Open-Sora Plan team moving forward.
