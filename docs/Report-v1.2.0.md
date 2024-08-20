# Report v1.2.0

In May 2024, we launched Open-Sora-Plan v1.1.0, featuring a 2+1D model architecture that could be quickly utilized for exploratory training in text-to-video generation tasks. However, when handling dense visual tokens, the 2+1D architecture could not simultaneously process spatial and temporal dimensions. Therefore, we transitioned to **a 3D full attention architecture**, which better captures the joint spatial-temporal features. Although this version is experimental, it advances video generation architecture to a new realm, leading us to release it as v1.2.0.

Compared to previous video generation models, Open-Sora-Plan v1.2.0 offers the following improvements:

1. **Better compressed visual representations**. We optimized the structure of CausalVideoVAE, which now delivers enhanced performance and higher inference efficiency.
2. **Better video generation architecture**. Instead of 2+1D, we use a diffusion model with a 3D full attention architecture, which provides a better understanding of the world.


### Open-Source Release
We open-source the Open-Sora-Plan to facilitate future development of Video Generation in the community. Code, data, model are made publicly available.
- Code: All training scripts and sample scripts.
- Model: Both Diffusion Model and CausalVideoVAE [here](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0).
- Data: Filtered data [here](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0).


## Gallery

93×1280×720 Text-to-Video Generation. The video quality has been compressed for playback on GitHub.

<table class="center">
<tr>
  <td><video src="https://github.com/user-attachments/assets/1c84bc92-d585-46c9-ae7c-e5f79cefea88" autoplay></td>
</tr>
</table>

## Detailed Technical Report

### CausalVideoVAE

#### Model Structure

The VAE in version 1.2.0 maintains the overall architecture of the previous version but merges the temporal and spatial downsampling layers. In version 1.1.0, we performed spatial downsampling (stride=1,2,2) followed by temporal downsampling (stride=2,1,1). In version 1.2.0, we conduct both spatial and temporal downsampling simultaneously (stride=2,2,2) and perform spatial-temporal upsampling in the decoder (interpolate_factor=2,2,2).

Due to the absence of additional convolutions during downsampling and upsampling, this method more seamlessly inherits the weights from the SD2.1 VAE, leading to improved initialization of our VAE.

<img src="https://s21.ax1x.com/2024/07/24/pkHrHx0.png" width=768>


#### Training Details


As with v1.1.0, we initialize from the [SD2.1 VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse) using tail initialization. We perform the first phase of training on the Kinetic400 video dataset, then use the EMA weights from this phase to initialize the second phase, which is fine-tuned on high-quality data (collected in v1.1.0). All training is conducted on 25-frame 256×256 videos using **one A100 node**.


| Training stage | Dataset | Training steps  |
|---|---|---|
| 1 |  K400 |  200,000 |
| 2 |  collected in v1.1.0 |  450,000   |


#### Evaluation

We evaluated our VAE on the validation sets of two video datasets: [Webvid](https://github.com/m-bain/webvid) and [Panda70m](https://github.com/snap-research/Panda-70M/), and compared it with our [v1.1.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.1.0.md), [SD2.1 VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse), [CV-VAE](https://github.com/AILab-CVC/CV-VAE), and [Open-Sora's VAE](https://github.com/hpcaitech/Open-Sora). The Webvid validation set contains 5k videos, while the Panda70m validation set has 6k videos. The videos were resized to 256 pixels on the short side, center-cropped to 256x256, and then 33 consecutive frames were extracted. We used PSNR, SSIM, and LPIPS metrics, and measured the encoding speed on an A100 GPU. The specific results are as follows:


**WebVid**

| Model | Compress Ratio |PNSR↑ | SSIM↑ |LPIPS↓ |
|---|---|---|---|---|
| SD2-1 VAE | 1x8x8 | 30.19 | 0.8379 | 0.0568 |
| SVD VAE | 1x8x8 |<ins>31.15</ins> |<ins>0.8686</ins> | **0.0547** | 
| CV-VAE | 4x8x8 | 30.76 | 0.8566 | 0.0803 |
| Open-Sora VAE | 4x8x8 | 31.12 | 0.8569 | 0.1003 |
|  Open-Sora Plan v1.1 | 4x8x8 | 30.26 | 0.8597 |<ins>0.0551</ins> |
|  Open-Sora Plan v1.2  | 4x8x8| **31.16** | **0.8694** | 0.0586 |

**Panda70M**

| Model | Compress Ratio| PNSR↑ | SSIM↑ |LPIPS↓ |
|---|---|---|---|---|
| SD2-1 VAE | 1x8x8 |30.40 | 0.8894 | 0.0396 |
| SVD VAE | 1x8x8 |<ins>31.00</ins> | **0.9058** | **0.0379** | 
| CV-VAE  | 4x8x8| 29.57 | 0.8795 | 0.0673 |
| Open-Sora VAE | 4x8x8 | **31.06** | 0.8969 | 0.0666 |
| Open-Sora Plan v1.1 | 4x8x8 | 29.16 | 0.8844 | 0.0481 |
|  Open-Sora Plan v1.2  | 4x8x8| 30.49 |<ins>0.8970</ins> |<ins>0.0454</ins>|

**Encode Time on A100**

|Input Size| CV-VAE | Open-Sora | Open-Sora Plan v1.1 | Open-Sora Plan v1.2 |
|---|---|---|---|---|
| 33x256x256 | 0.186 | 0.147 |<ins>0.104</ins> | **0.102** |
| 81x256x256 | 0.465 | 0.357 |<ins>0.243</ins> | **0.242** |

### Training Text-to-Video Diffusion Model

#### Model Structure

The most significant change is that we **replaced all 2+1D Transformer blocks with 3D full attention blocks**. Each video is first processed by a patch embedding layer, which downsamples the spatial dimensions by a factor of 2. The video is then flattened into a one-dimensional sequence across the frame, width, and height dimensions. We replaced [T5-XXL](https://huggingface.co/DeepFloyd/t5-v1_1-xxl) with [mT5-XXL](https://huggingface.co/google/mt5-xxl) to enhance multilingual adaptation. Additionally, we incorporated RoPE.


### Sequence Parallelism

Due to the high computational complexity of 3D full attention, we must allocate a video across 2 GPUs for parallel processing when training with long-duration and high-resolution videos. We can control the number of GPUs used for a video sample by adjusting the batch size on a node. For example, with `sp_size=8` and `train_sp_batch_size=4`, 2 GPUs are used for a single sample. **We support sequence parallelism for both training and inference**.

**Training on 93×720p**, we report speed on H100.

| GPU （sp_size） | batch size | Enable sp | Train_sp_batch_size | Speed | Step per day |
|---|---|---|---|---|---|
|8|8|×|-|100s/step|~850|
|8|-|√|4|53s/step|~1600|
|8|-|√|2|27s/step|~3200|

**Inference on 93×720p**, we report speed on H100.

| Size | 1 GPU | 8 GPUs | 
|---|---|---|
|29×720p|420s/100step|80s/100step|
|93×720p|3400s/100step|450s/100step|

#### Dynamic training

Deep neural networks are typically trained using batched inputs. For efficient hardware processing, batch shapes are fixed, leading to a fixed data size. This requires either cropping or padding images to a uniform size, both of which have drawbacks: cropping degrades performance, while padding is inefficient and results in significant information loss. Generally, there are three methods for training with arbitrary token counts: Patch n' Pack, bucket, and pad-mask.



**Patch n' Pack** ([NaViT](https://arxiv.org/abs/2307.06304)): bypasses the fixed sequence length limitation by combining tokens from multiple samples into a new sample. This approach allows variable-resolution images while maintaining aspect ratios by packaging multiple samples together, thereby reducing training time and enhancing performance and flexibility. However, this method involves significant code modifications and requires re-adaptation when exploring different model architectures in fields with unstable model designs.


**Bucket** ([Pixart-alpha](https://arxiv.org/abs/2310.00426), [Open-Sora](https://github.com/hpcaitech/Open-Sora)): This method packages data of different resolutions into buckets, sampling batches from each bucket to ensure same resolution within each batch. It requires minimal code modifications to the model, mainly adjusting the data sampling strategy.

**Pad-mask** ([FiT](https://arxiv.org/abs/2402.12376), our v1.0/v1.1): This method sets a maximum resolution and pads all data to this resolution, generating a corresponding mask. Although the approach is straightforward, it is computationally inefficient.

We believe that current video generation models are still in an exploratory phase. Extensive modifications to model code during this period can incur unnecessary development costs. The pad-mask method, while straightforward, is computationally inefficient and can waste resources in video, which involves dense computations. Ultimately, we chose the bucket strategy, which requires no modifications to the model code. Next, we will explain how our bucket strategy supports arbitrary lengths and resolutions. For simplicity, we will use video duration as an example:

<img src="https://s21.ax1x.com/2024/07/24/pkHr4aQ.png" width=768>


We define a megabatch as the total data processed in a single step across all GPUs. A megabatch can be divided into multiple batches, with each batch corresponding to the data processed by a single GPU.

**Sort by frame**: The first step is to count the number of frames in all video data and sort them. This step aims to group similar data together, with sorting being one method to achieve this.

**Group megabatch**: Next, all data is divided into groups, each forming a megabatch. Since all data is pre-sorted, most videos within a megabatch have the same number of frames. However, there will always be boundary cases, such as having both 61-frame and 1-frame videos in a single megabatch.

**Re-organize megabatch**: We re-organize these special megabatches, which actually constitute a small proportion. We randomly replace the minority data in the megabatch with the majority data, thus re-organizing it into a megabatch with same frame counts.

**Shuffle megabatch**: To ensure data randomness, we shuffle both within each megabatch and between different megabatches.

When supporting dynamic resolutions, we simply replace each sample's frame sequence with (frame × height × width). This method ensures that the data dimension processed by each GPU in every step is the same, preventing situations where GPU1 waits for GPU0 to finish processing a longer video. Moreover, it is entirely decoupled from the model code, serving as a plug-and-play video sampling strategy.


#### Training stage

Similar to previous work, we use a multi-stage training approach. With the 3D DiT architecture, all parameters can be transferred from images to videos without loss. To explore training costs, all parameters of the diffusion model are trained from scratch. Therefore, we first train an text-to-image model, using the training strategy from [Pixart-alpha](https://arxiv.org/abs/2310.00426).

The video model is initialized with weights from a 480p image model. We first train 480p videos with 29 frames. Next, we adapt the weights to 720p resolution, training on approximately 6 million higher-quality (HQ) samples from Panda70M, filtered for aesthetic quality and motion. Finally, we refine the model with a more higher-quality (HQ) subset of 1 million samples. After that, we use a filtered data (collected in v1.1.0) for fine-tuning 93-frame 720p videos. Below is our training card. We release the annotation file [here](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/anno_json).

| Name | Stage 1 | Stage 2 | Stage 3 | Stage 4 |Stage 5 |
|---|---|---|---|---|---|
| Training Video Size | 1×320×240 |  1×640×480 | 29×640×480 |  29×1280×720 | 93×1280×720 |
| Training Step| 146k |  200k | 30k | 21k | 3k |
| Compute (#Num x #Hours) | 32 Ascend × 81 | 32 Ascend × 142 |  128 Ascend × 38 | 256 H100 × 64 | 256 H100 × 84 |
| Checkpoint | - | [HF](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/1x480p) | [HF](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x480p) | [HF](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x720p) | [HF](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x720p) |
| Log | - | - | [wandb](https://api.wandb.ai/links/1471742727-Huawei/trdu2kba) | [wandb](https://api.wandb.ai/links/linbin/vvxvcd7s) | [wandb](https://api.wandb.ai/links/linbin/easg3qkl)
| Training Data | [10M SAM](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/blob/main/anno_json/sam_image_11185255_resolution.json) | 5M internal image data | [6M HQ Panda70M](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/blob/main/anno_json/Panda70M_HQ6M.json) | [6M HQ Panda70M](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/blob/main/anno_json/Panda70M_HQ6M.json) | [1M HQ Panda70M](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/blob/main/anno_json/Panda70M_HQ1M.json) and [100k HQ data](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/anno_json) (collected in v1.1.0) |

Additionally, we fine-tuned 3.5k steps from the final 93×720p to get [93×480p](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/93x480p) for community research use.

### Training Image-to-Video Diffusion Model

#### Model Structure

<img src="https://s21.ax1x.com/2024/08/12/pApZZJf.png">

To reuse the weights of the Text-to-Video model, our Image-to-Video model is inspired by the Stable Diffusion Inpainting Model and adopts a strategy based on frame-level inpainting. By incorporating three types of information—original noise, masked video, and mask—under different control frame conditions, our model can generate coherent videos while ensuring flexibility in its usage.

Compared to the denoiser structure of the Text-to-Video model, the Inpainting Model's denoiser has only changed the number of channels in the `conv in` layer. To ensure the model has a good prior knowledge, we introduce the masked video and mask information through zero initialization. We believe this is due to the 2+1D structure's lack of ability to establish long-range information dependencies, and relying solely on attention in the temporal dimension makes it difficult to capture information changes under frame control. In Text-to-Video tasks, this phenomenon is not as evident because all frames share the same text prompt embedding. However, in Image-to-Video tasks, simply concatenating images in the channel dimension does not ensure the model can accurately capture changes between frames. This is because the model cannot directly replicate image information from the channels to reduce the loss, and the 2+1D structure's interaction solely on the temporal axis fails to allow the model to discern which information from the control frames can be utilized, especially there are significant differences between frames. Therefore, without a shared image-semantic information, the control frame information might not be effectively conveyed to each frame.

##### About Semantic Adapter

In previous models based on the Unet 2+1D architecture, it is necessary to input the control frames into the CLIP model to obtain semantic embeddings. These semantic embeddings are then injected into the denoiser through cross-attention. The structure that extracts CLIP embeddings and injects them into the denoiser is commonly referred to as a semantic adapter.

In the 2+1D architecture, the semantic adapter is commonly present. Additionally, papers like [DynamiCrafter](https://arxiv.org/abs/2310.12190) have pointed out that incorporating the semantic adapter helps maintain stability in the generated videos. We believe this is because the 2+1D structure lacks the ability to establish long-range information dependencies, and relying solely on attention in the temporal dimension makes it difficult to capture information changes under frame control. In the Text-to-Video task, this phenomenon is not as evident because all frames share the same text prompt embedding. However, in the Image-to-Video task, without shared semantic information, it may lead to the inability to effectively transfer control frame information to each individual frame.

<center>
<figure>
    <img src="https://github.com/user-attachments/assets/06df193a-fe89-42c1-8c01-fb3b7c2be0e3" height=400 />
	<img src="https://github.com/user-attachments/assets/09906df4-ab9f-443d-8d38-512e16075b0c" height=400 />
</figure>
</center>

We conducted a simple comparison of the performance of using the Inpainting Model under the 2+1D structure (Open-Sora Plan v1.1, left in the figure) versus the 3D structure (Open-Sora Plan v1.2, right in the figure). With the same number of optimization steps, the probability of unstable visual performance in the 2+1D structure was significantly higher than in the 3D structure. Even at convergence, the 2+1D structure's visual stability was still inferior to that of the 3D structure, and it was even worse than the early training stages of the 3D structure.

## Future Work and Discussion

#### CausalVideoVAE
We observed that high-frequency motion information in videos tends to exhibit jitter, and increasing training duration and data volume does not significantly alleviate this issue. In videos, compressing the duration while maintaining the original latent dimension can lead to significant information loss. A more robust VAE will be released in the next version.

#### Diffusion Model
We replaced T5 with mT5 to enhance multilingual capabilities, but this capability is limited as our training data is currently only in English. The multilingual ability primarily comes from the mT5 mapping space. We will explore additional text encoders and expand the data in the next steps.

Our model performs well in generating character consistency, likely due to panda70m being a character-centric dataset. However, it still shows poor performance in text consistency and object generalization. We suspect this may be due to the limited amount of data the model has seen, as evidenced by the non-convergence of the loss in the final stage. **We hope to collaborate with the open-source community to optimize the 3D DiT architecture.**
