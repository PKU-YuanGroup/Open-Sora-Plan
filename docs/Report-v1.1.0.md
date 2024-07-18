# Report v1.1.0

In April 2024, we launched Open-Sora-Plan v1.0.0, featuring a simple and efficient design along with remarkable performance in text-to-video generation. It has already been adopted as a foundational model in numerous research projects, including its data and model.

**Today, we are excited to present Open-Sora-Plan v1.1.0, which significantly improves video generation quality and duration.**

Compared to the previous version, Open-Sora-Plan v1.1.0, the improvements include:

1. **Better compressed visual representations**. We optimized the CausalVideoVAE architecture, which now has stronger performance and higher inference efficiency.
2. **Generate higher quality, longer videos**. We used higher quality visual data and captions by [ShareGPT4Video](https://sharegpt4video.github.io/), enabling the model to better understand the workings of the world.

Along with performance improvements, Open-Sora-Plan v1.1.0 maintains the minimalist design and data efficiency of v1.0.0. Remarkably, we found that v1.1.0 exhibits similar performance to the Sora base model, indicating that our version's evolution aligns with the scaling law demonstrated by Sora.

### Open-Source Release
We open-source the Open-Sora-Plan to facilitate future development of Video Generation in the community. Code, data, model will be made publicly available.
- Demo: Hugging Face demo [here](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.1.0).
- Code: All training scripts and sample scripts.
- Model: Both Diffusion Model and CasualVideoVAE [here](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0).
- Data: Both raw videos and captions [here](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0).

## Gallery


### 221×512×512 Text-to-Video Generation

| 221×512×512 (9.2s) | 221×512×512 (9.2s) | 221×512×512 (9.2s) | 221×512×512 (9.2s) |
| --- | --- | --- | --- |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/6d18f344-f7da-44eb-9e07-77813f6b5e90" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/71f75e72-e9ee-4ce7-b8ea-d2d45a6f367e" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/80430eae-a3b4-4f24-b448-0db2919327d6" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/dc217894-a8c8-4174-a42c-acd2811f61f5" width=224> |
| This close-up shot of a Victoria crowned pigeon showcases its striking blue plumage ... |  a cat wearing sunglasses and working as a lifeguard at pool. |  Photorealistic closeup video of two pirate ships battling each other as they sail ... | A movie trailer featuring the adventures ofthe 30 year old spacemanwearing a redwool ...  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/f9c5823f-aa03-40ee-8335-684684f5c842" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/f207d798-8988-45b0-b836-347e499ee000" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/d40c38dc-9f26-4591-8163-c7089e1553e3" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/67b40ce1-135d-4ae2-a1ee-5a0866461eb2" width=224> |
| A snowy forest landscape with a dirt road running through it. The road is flanked by ... |  Drone shot along the Hawaii jungle coastline, sunny day. Kayaks in the water. | Alpacas wearing knit wool sweaters, graffiti background, sunglasses.  | The camera rotates around a large stack of vintage televisions all showing different ...  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/42542b4e-b1b8-49b8-ada4-7bcfde8e1453" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/4eee2619-a5ca-4a32-b350-e65d6220c8f7" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/e6152c93-4edf-4569-8d17-1effd87a7780" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/ec78d855-07c7-4421-896a-3c46a83ec129" width=224> |
| A drone camera circles around a beautiful historic church built on a rocky outcropping ... | Aerial view of Santorini during the blue hour, showcasing the stunning architecture ...  |  A robot dog explores the surface of Mars, kicking up red dust as it investigates  ... | An aerial shot of a lighthouse standing tall on a rocky cliff, its beacon cutting ...  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/98897215-eae9-49f3-8fdb-df1d4b74d435" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/10e0f93d-925f-4b38-8205-7d89f49195f1" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/5453cf37-29ac-423d-9fb2-05f23416ca3e" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/77928824-6705-4d83-a7b4-43fdb74790bf" width=224> |
| 3D animation of a small, round, fluffy creature with big, expressive eyes explores ... |  A corgi vlogging itself in tropical Maui. |  A single drop of liquid metal falls from a floating orb, landing on a mirror-like ... | The video presents an abstract composition centered around a hexagonal shape adorned ...  |

### 65×512×512 Text-to-Video Generation

| 65×512×512 (2.7s) | 65×512×512 (2.7s) | 65×512×512 (2.7s) | 65×512×512 (2.7s) |
| --- | --- | --- | --- |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/a0601f20-579c-4e2e-832c-5763546718cc" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/55229eca-de3a-476b-930b-13a35eb5db30" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/74b9e7a1-0fa4-4f0d-8faf-0c84d97b11b5" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/66b06822-2652-453e-80fb-dd2988b730ce" width=224> |
| Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. | 3D animation of a small, round, fluffy creature with big, expressive eyes explores a ...   |  A corgi vlogging itself in tropical Maui. |  In a studio, there is a painting depicting a ship sailing through the rough sea. |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/4d27ff13-e725-4602-bf17-90df2c0d8005" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/049fd4db-f2fe-4633-ab62-1dda8268e090" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/aba01259-60f2-49ef-aa33-e738dc8c9a49" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/98178397-61b0-4200-9ab9-9bef2728ed98" width=224> |
| A robot dog trots down a deserted alley at night, its metallic paws clinking softly ... | A solitary spider weaves its web in a quiet corner. The web shimmers and glows with ...  |  A lone surfer rides a massive wave, skillfully maneuvering through the surf. The water ... |  A solitary cheetah sprints across the savannah, its powerful muscles propelling it ... |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/3964dfcd-d1b4-406b-916c-d4a702184a27" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/2a5fd1c0-9304-46e4-af35-5ffcc718bf08" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/73e1304a-6241-4e19-9dde-f2b1032edefd" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/c82b822b-cadd-4f11-bcc3-29e3651c02c0" width=224> |
| A solitary astronaut plants a flag on an alien planet covered in crystal formations ... |  At dawn's first light, a spaceship slowly exits the edge of the galaxy against a ...|  A dapper puppy in a miniature suit, basking in the afternoon sun, adjusting his tie ... |  A wise old elephant painting abstract art with its trunk, each stroke a burst of color ... |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/8edfa249-272c-4773-8728-12686527771e" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/33338d8e-5ea7-4b57-9e94-97f3ee404033" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/227fa6b6-801b-438b-9b30-cd3a4e0a7f2f" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/b5d5ebee-ab13-4e32-8b79-e6cb8f08d4a0" width=224> |
| In an ornate, historical hall, a massive tidal wave peaks and begins to crash. Two ... | A Shiba Inu dog wearing a beret and black turtleneck.  | A painting of a boat on water comes to life, with waves crashing and the boat becoming ...  | Many spotted jellyfish pulsating under water. Their bodies are transparent and glowing ...  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/fe8e9450-5a80-435d-b050-6b2fe11cdf53" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/7d9335df-817d-479d-9ea7-615cb50c66b8" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/34c1dda6-e420-4edd-bbcd-38ff8e542ec6" width=224> |  |
| An animated hedgehog with distinctive spiky hair and large eyes is seen exploring a ... | An animated rabbit in a playful pink snowboarding outfit is carving its way down a ...  | A person clad in a space suit with a helmet and equipped with a chest light and arm ...  |   |

### 65×512×512 Video Editing

| generated 65×512×512 (2.7s) | edited 65×512×512 (2.7s) |
| --- | --- |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/edb8e8c2-5eef-4c90-85fb-6adb035067c3" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/32d93845-8904-4f8f-832e-37eba2ceb542" width=224>  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/a70d0d91-0d61-4aa4-9520-6e4a6c477f12" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/7913e06f-1e0b-4d06-8233-72c882c6abfe" width=224>  |
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/0e614ec0-fba0-4f42-a343-d4607966dd40" width=224> | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/c531d930-410c-4614-890d-bae8013f33c2" width=224>  |

### 512×512 Text-to-Image Generation

 <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/e44b7f8a-5da2-49c2-87c4-52ea680ad43b" width=512> 

## Detailed Technical Report

### CasualVideoVAE

#### Model Structure

As the number of frames increases, the encoder overhead of CausalVideoVAE gradually rises. When training with 257 frames, 80GB of VRAM is insufficient for the VAE to encode the video. Therefore, we reduced the number of CausalConv3D layers, retaining only the last two stages of CausalConv3D in the encoder. This change significantly lowers the overhead while maintaining nearly the same performance. Note that we only modified the encoder; the decoder still retains all CausalConv3D layers, as training the Diffusion Model does not require the decoder.

<img width="722" alt="vaemodel" src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/b6f8a638-cfb2-40f3-94be-45af8dbad18e">

We compare the computational overhead of the two versions by testing the forward inference of the encoder on the H100.

| Version | 129×256×256 |   | 257×256×256 | | 513×256×256 | |
|---|---|---|---|---|---|---|
|  |  Peak Mem. |  Speed  | Peak Mem. |  Speed  |Peak Mem. |  Speed  |
| v1.0.0 |  22G |   2.9 it/s  | OOM |  -   | OOM |   -  |
| v1.1.0 |  18G |  4.9 it/s   | 34G  |  2.5 it/s   | 61G |   1.2 it/s   |


#### Temporal Module

<img width="480" alt="vaemodel" src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/7c8a4263-b7d1-4edc-a60c-801ae9b4344f">

In v1.0.0, our temporal module had only TemporalAvgPool. TemporalAvgPool leads to the loss of high-frequency information in the video, such as details and edges. To address this issue, we improved this module in v1.1.0. As shown in the figure below, we introduced convolution and added learnable weights, allowing different branches to decouple different features. When we omit CausalConv3D, the video is reconstructed very blurry. Similarly, when we omit TemporalAvgPool, the video becomes very sharp.

|  | SSIM↑ | LPIPS↓ | PSNR↑ |
|---|---|---|---|
| Base | 0.850 |  0.091 |  28.047 |
| + Frames | 0.868 |  0.070 |  28.829 | 
| + Reset mixed factor | 0.873 |  0.070 |  29.140 | 





#### Training Details

Similar to v1.0.0, we initialized from the Latent Diffusion's VAE and used tail initialization. For CausalVideoVAE, we trained for 100k steps in the first stage with a video shape of 9×256×256. Subsequently, we increased the frame count from 9 to 25 and found that this significantly improved the model's performance. It is important to clarify that we enabled the mixed factor during both the first and second stages, with a value of a (sigmoid(mixed factor)) reaching 0.88 at the end of training, indicating the model's tendency to retain low-frequency information. In the third stage, we reinitialized the mixed factor to 0.5 (sigmoid(0.5) = 0.6225), which further enhanced the model's capabilities.

#### Loss Function

We found that using GAN loss helps retain high-frequency information and alleviates grid artifacts. Additionally, we observed that switching from 2D GAN to 3D GAN provides further improvements.

| GAN Loss/Step | SSIM↑ | LPIPS↓ | PSNR↑ |
|---|---|---|---|
| 2D/80k | 0.879 |  0.068 |  29.480 |
| 3D/80k | 0.882 |  0.067 |  29.890 | 

#### Inference Tricks
Therefore, we introduced a method called **temporal rollback tiled convolution**, a tiling approach specifically designed for CausalVideoVAE. Specifically, all windows except the first one discard the first frame because the first frame in a window is treated as an image, while the remaining frames should be treated as video frames.

<img width="633" alt="tiled_temp" src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/0a06011e-1d6c-410a-9f1c-82c4122a018a">

We tested the speed on the H100 with a window size of 65×256×256.

| Version | 129×256×256 |   | 257×256×256 | | 513×256×256 | |
|---|---|---|---|---|---|---|
|  |  Peak Mem. |  Speed  | Peak Mem. |  Speed  |Peak Mem. |  Speed  |
| 4×8×8 |  10G |   1.3 s/it  | 10G | 2.6 s/it   | 10G |   5.3 s/it  |

### Data Construction
Since Open-Sora-Plan supports joint training of images and videos, our data collection is divided into two parts: images and videos. Images do not need to originate from videos; they are independent datasets. We spent approximately 32×240 H100 hours generating image and video captions, and all of this is **open source**!

#### Image-Text Collection Pipeline
We obtained 11 million image-text pairs from [Pixart-Alpha](https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M), with captions generated by [LLaVA](https://github.com/haotian-liu/LLaVA). Additionally, we utilized the high-quality OCR dataset [Anytext-3M](https://github.com/tyxsspa/AnyText), which pairs each image with corresponding OCR characters. However, these captions were insufficient to describe the entire image, so we used [InternVL-1.5](https://github.com/OpenGVLab/InternVL) for supplementary descriptions. Since T5 only supports English, we filtered for English data, which constitutes about half of the complete dataset. Furthermore, we selected high-quality images from [Laion-5B](https://laion.ai/blog/laion-5b/) to enhance human-like generation quality. The selection criteria included high resolution, high aesthetic scores, and watermark-free images containing people.

Here, we are open-sourcing the prompt used for InternVL-1.5:
```
# for anytext-3m
Combine this rough caption: "{}", analyze the image in a comprehensive and detailed manner. "{}" can be recognized in the image.
# for human-160k
Analyze the image in a comprehensive and detailed manner.
```

| Name | Image Source | Text Captioner | Num pair |
|---|---|---|---|
| SAM-11M | [SAM](https://ai.meta.com/datasets/segment-anything/) |  [LLaVA](https://github.com/haotian-liu/LLaVA) |  11,185,255 |
| Anytext-3M-en | [Anytext](https://github.com/tyxsspa/AnyText) |  [InternVL-1.5](https://github.com/OpenGVLab/InternVL) |  1,886,137 | 
| Human-160k | [Laion](https://laion.ai/blog/laion-5b/) |  [InternVL-1.5](https://github.com/OpenGVLab/InternVL) |  162,094 | 


#### Video-Text Collection Pipeline
In v1.0.0, we sampled one frame from each video to generate captions. However, as video length increased, a single frame could not adequately describe the entire video's content or temporal movements. Therefore, we used a video captioner to generate captions for the entire video clip. Specifically, we used [ShareGPT4Video](https://sharegpt4video.github.io/), which effectively covers temporal information and describes the entire video content. The v1.1.0 video dataset comprises approximately 3k hours, compared to only 300 hours in v1.0.0. As before, we have open-sourced all text annotations and videos (both under the CC0 license), which can be found [here](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main).



| Name | Hours | Num frames | Num pair |
|---|---|---|---|
| [Mixkit](https://mixkit.co/) | 42.0h |  65 |  54,735 |
|   |  |  513 |  1,997 | 
| [Pixabay](https://pixabay.com/) | 353.3h |  65 | 601,513 |
|   |  |  513 |  51,483 | 
| [Pexel](https://www.pexels.com/) | 2561.9h |  65 |  3,832,666 |
|   |  |  513 |  271,782 | 

### Training Diffusion Model
Similar to our previous work, we employed a multi-stage cascaded training method. Below is our training card:

#### Stage 1

Surprisingly, we initially believed that the performance of the diffusion model would improve with longer training. However, by observing the [logs](https://api.wandb.ai/links/linbin/o76j03j4), we found that videos generated at 50k steps were of higher quality than those at 70-100k steps. In fact, extensive sampling revealed that checkpoints at 40-60k steps outperformed those at 80-100k steps. Quantitatively, 50k steps correspond to approximately 2 epochs of training. It is currently unclear whether this is due to overfitting from a small dataset or the limited capacity of the 2+1D model.

#### Stage 2

In the second stage, we used Huawei Ascend computing power for training. This stage's training and inference were fully supported by Huawei. We conducted sequence parallel training and inference on a large-scale cluster, distributing one sample across eight ranks. Models trained on Huawei Ascend can also be loaded into GPUs and generate videos of the same quality.


#### Stage 3

In the third stage, we further increased the frame count to 513 frames, approximately 21 seconds at 24 FPS. However, this stage presents several challenges, such as ensuring temporal consistency in the 2+1D model over long durations and whether the current amount of data is sufficient. We are still training the model for this stage and continuously monitoring its progress.

| Name | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Training Video Size | 65×512×512 |  221×512×512 | 513×512×512 |
| Compute (#Num x #Hours) | 80 H100 × 72 | 512 Ascend × 72 |  Under Training |
| Checkpoint | [HF](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/65x512x512) | [HF](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/221x512x512) | Under Training |
| Log | [wandb](https://api.wandb.ai/links/linbin/o76j03j4) | - |  - |
| Training Data | ~3k hours videos + 13M images |  |  |

### Video Editing

The recently proposed [ReVideo](https://mc-e.github.io/project/ReVideo/) achieves accurate video editing by modifying the first frame and applying motion control within the edited area. Although it achieves excellent video editing performance, the editing length is limited by the base model [SVD](https://github.com/Stability-AI/generative-models). Open-Sora, as a fundamental model for long-video generation, can compensate for this issue. Currently, we are collaborating with the ReVideo team to use Open-Sora as the base model for long video editing. Some preliminary results are shown [here](). 

The initial version still needs improvement in several aspects. In the future, we will continue to explore integration with ReVideo to develop improved long-video editing models.

## Failed Case and Discussion

Despite the promising results of v1.1.0, there remains a gap between our model and Sora. Here, we present some failure cases and discuss them.

### CasualVideoVAE

Despite the significant performance improvement of VAE in v1.1.0 over v1.0.0, we still encounter failures in challenging cases, such as sand dunes and leaves. The video on the left shows the reconstructed video downsampled by a factor of 4 in time, while the video on the right is downsampled by a factor of 2. Both exhibit jitter when reconstructing fine-grained features. This indicates that reducing temporal downsampling alone cannot fully resolve the jitter issue.

https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/1a87d6d8-4bf1-4b4e-83bb-84870c5c3a11

https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/1a87d6d8-4bf1-4b4e-83bb-84870c5c3a11

### Diffusion Model

#### Semantic distortion

On the left is a video generated by v1.1.0 showing a puppy in the snow. In this video, the puppy's head exhibits semantic distortion, indicating that the model struggles to correctly identify which head belongs to which dog. On the right is a video generated by Sora's [base model](https://openai.com/index/video-generation-models-as-world-simulators/). We observe that Sora's early base model also experienced semantic distortion issues. This suggests that we may achieve better results by scaling up the model and increasing the amount of training data.

Prompt：A litter of golden retriever puppies playing in the snow.Their heads pop out of the snow, covered in.

| Our | Sora Base×1 | Sora Base×4 | Sora Base×32 |
|---|---|---|---|
| <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/1d456168-afad-4e22-ae3b-fc28eca935e8" width=224>  |<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/c4ca99d9-9492-45c8-a75e-6efe21c330aa" width=224>  |<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/b7b52894-f58b-4e64-858b-015247108b8b" width=224>  | <img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/dcaf793b-1da4-4cc1-9c17-8a46d55e80e6" width=224> |

#### Limited dynamics

The primary difference between videos and images lies in their dynamic nature, where objects undergo a series of changes across consecutive frames. However, the videos generated by v1.1.0 still contain many instances of limited dynamics. Upon reviewing a large number of training videos, we found that while web-crawled videos have high visual quality, they are often filled with meaningless close-up shots. These close-ups typically show minimal movement or are even static. On the left, we present a generated video of a bird, while on the right is a training video we found, which is almost static. There are many similar videos in the dataset from stock footage sites.

Prompt：This close-up shot of a Victoria crowned pigeon showcases its striking blue plumage and red chest. Its crest is made of delicate, lacy feathers, while its eye is a striking red color. The bird's head is tilted slightly to the side,giving the impression of it looking regal and majestic. The background is blurred,drawing attention to the bird's striking appearance.


| Our | Raw video |
|---|---|
|<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/7ffb6bc6-b52c-488e-9f29-d7d90bda44d6" width=224>  |<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/95fbbca4-e206-42f6-8c6b-af1063c442c6" width=224>  | 

#### Negative prompt

We found that using negative prompts can significantly improve video quality, even though we did not explicitly tag the training data with different labels. On the left is a video sampled using a negative prompt, while on the right is a video generated without a negative prompt. This suggests that we may need to incorporate more prior knowledge into the training data. For example, when a video has a watermark, we should note "watermark" in the corresponding caption. When a video's bitrate is too low, we should add more tags to distinguish it from high-quality videos, such as "low quality" or "blurry." We believe that explicitly injecting these priors can help the model differentiate between the vast amounts of pretraining data (low quality) and the smaller amounts of fine-tuning data (high quality), thereby generating higher quality videos.

Prompt：A litter of golden retriever puppies playing in the snow.Their heads pop out of the snow, covered in.
Negative Prompt：distorted, discontinuous, ugly, blurry, low resolution, motionless, static, low quality


| With Negative Prompt | Without Negative Prompt |
|---|---|
|<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/1d456168-afad-4e22-ae3b-fc28eca935e8" width=224>  |<img src="https://github.com/PKU-YuanGroup/Open-Sora-Plan/assets/62638829/7ad17f96-bfab-455a-830d-0daebccaf6fb" width=224>  | 

## Future Work

In our future work, we will focus on two main areas: (1) data scaling and (2) model design. Once we have a robust baseline model, we will extend it to handle variable durations and conditional control models.

### Data Scaling

#### Data source

As mentioned earlier, our dataset is entirely sourced from stock footage websites. Although these videos are of high quality, many consist of close-up shots of specific areas, resulting in slow motion in the videos. We believe this is one of the main reasons for the limited dynamics observed. Therefore, we will continue to collect datasets from diverse sources to address this issue.

#### Data volume

In v1.1.0, our dataset comprises only ~3k hours of video. We are actively collecting more data and anticipate that the video dataset for the next version will reach ~100k hours. We welcome recommendations from the open-source community for additional datasets.

### Model Design

#### CasualVideoVAE
In our internal testing, even without downsampling in time, we found that it is not possible to completely resolve the jitter issue in reconstructing fine-grained features. Therefore, we need to reconsider how to mitigate video jitter to the greatest extent possible while simultaneously supporting both images and videos. We will introduce a more powerful CasualVideoVAE in the next version.

#### Diffusion Model
In v1.1.0, we found that 2+1D models can generate higher-quality videos in short durations. However, for long videos, they tend to exhibit discontinuities and inconsistencies. Therefore, we will explore more possibilities in model architecture to address this issue.
