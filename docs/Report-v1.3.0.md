# Report v1.3.0

In August 2024, we released Open-Sora-Plan v1.2.0, transitioning to a 3D full attention architecture, which enhanced the capture of joint spatial-temporal features. However, the substantial computational cost made it unsustainable, and the lack of a clear training strategy hindered continuous progress along a focused path.

In version 1.3.0, Open-Sora-Plan introduced the following five key features:

**1. A more powerful and cost-efficient WFVAE.** We decompose video into several sub-bands using wavelet transforms, naturally capturing information across different frequency domains, leading to more efficient and robust VAE learning.

**2. Prompt Refiner.** A large language model designed to refine short text inputs.

**3. High-quality data cleaning strategy.** The cleaned panda70m dataset retains only 27% of the original data.

**4. DiT with new sparse attention.** A more cost-effective and efficient learning approach.

**5. Dynamic resolution and dynamic duration.** This enables more efficient utilization of videos with varying lengths (treating a single frame as an image).

### Open-Source Release
We open-source the Open-Sora-Plan to facilitate future development of Video Generation in the community. Code, data, model will be made publicly available.
- Code: All training scripts and sample scripts.
- Model: Both Diffusion Model and CasualVideoVAE [here](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0).

## Gallery

Text & Image to Video Generation. 

[![Demo Video of Open-Sora Plan V1.3](https://github.com/user-attachments/assets/4ff1d873-3dde-4905-a907-dbff51174c20)](https://www.bilibili.com/video/BV1KR2fYPEF5/?spm_id_from=333.999.0.0&vd_source=cfda99203e659100629b465161f1d87d)

## Detailed Technical Report

### WF-VAE

As video generation models move toward higher resolutions and longer durations, the computational cost of video VAEs grows exponentially, becoming unsustainable. Most related work addresses this by using tiling to reduce inference memory consumption. However, in high-resolution, long-duration scenarios, tiling significantly increases inference time. Additionally, since tiling is lossy for latents, it can lead to visual artifacts such as shadows or flickering in the generated videos. Then, we introduce WFVAE, which provide a new model to handle these problems.

#### Model Structure

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/a107b046-d7e1-4a32-a429-0a061a4f8ee8" height=400 />
</figure>
</center>

The compression rate fundamentally determines the quality of VAE-reconstructed videos. We analyzed the energy and entropy of different subbands obtained through wavelet transform and found that most of the energy in videos is concentrated in the low-frequency bands. Moreover, by replacing the `LLL` subband of the VAE-reconstructed video with the original video's `LLL` subband, we observed a significant improvement in the spatiotemporal quality of the videos.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/533666a6-05be-4584-8b14-86f01d0471dd" height=250 />
</figure>
</center>

In previous VAE architectures, the lack of a "highway" for transmitting the dominant energy during video compression meant that this pathway had to be gradually established during model training, leading to redundancy in model parameters and structure. Therefore, in our model design, we created a more efficient transmission path for the LLL subband energy, significantly simplifying the model architecture, reducing inference time, and lowering memory consumption.



#### Training Details

Additional training details and design insights will be provided in the paper.

#### Ablation Study

In our experiments, we used the K400 training and validation sets, conducted on 8xH100 GPUs. The latent dimension was fixed at 4. We observed that as model parameters increased, there was still room for improvement in reconstruction metrics. GroupNorm showed instability during training, performing worse than LayerNorm on PSNR but better on LPIPS. Additional experiments will be detailed in the paper.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/ed880143-72d1-4316-a1d4-5fdfc5ed155a" height=200 />
	<img src="https://github.com/user-attachments/assets/303954c3-73ee-44f3-9897-d3d14b37b27e" height=200 />
</figure>
</center>


#### Performance


In the inference performance test on 33xPxP videos without tiling, WF-VAE significantly outperformed other open-source VAEs. 

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/bd2502d2-206c-4267-a769-98d1239b7f48" height=250 />
</figure>
</center>


During DiT training, we focused only on the encoding time and memory requirements of the VAE's encoder. We tested the encoding performance in float32 on H100.



**33x256x256**
| Model | Encode Memory | Encode Time | 
|---|---|---|
| WF-VAE-S|1614.00 MB |0.027 |
| OD-VAE | 8718.00 MB |0.088  |

**33x512x512**
| Model | Encode Memory | Encode Time | 
|---|---|---|
| WF-VAE-S | 4652.00 MB | 0.095 |
| OD-VAE | 31944.00 MB |0.353  |

**33x768x768**
| Model | Encode Memory | Encode Time | 
|---|---|---|
| WF-VAE-S | 9734.00 MB | 0.244 |
| OD-VAE | 65912.00 MB | 0.811 |

**93x768x768**
| Model | Encode Memory | Encode Time | 
|---|---|---|
| WF-VAE-S | 26382.00 MB | 0.668 |
| WF-VAE-M | 31438.00 MB | 0.841 |
| WF-VAE-L | 37098.00 MB | 1.216 |
| OD-VAE | OOM | N/A |

#### Evaluation

We evaluated PSNR and LPIPS on the Panda70M test set at 256 pixels and 33 frames. In the open-source WF-VAE-S (8-dim), our encoder was distilled from the 8-dim OD-VAE, resulting in some metric degradation compared to direct training.


| Latent Dim | Model | Params |  PSNR |  LPIPS | 
|---|---|---|---|---|
| 4 | OD-VAE（Our VAE in v1.2.0） | 94M + 144M | 30.311| 0.043|
| 4 | WFVAE-S | 38M + 108M |30.824 |0.052 |
| 8 | WFVAE-S（Distillion） |38M + 108M | 31.764|0.050 |

#### Causal Cache

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/59cb0543-225b-45a3-a4a6-429e5e753167" height=200 />
</figure>
</center>


To address the issue of tiling, we replaced GroupNorm with LayerNorm and introduced a novel method called **Causal Cache**, enabling lossless temporal block-wise inference.

First, we replaced GroupNorm with LayerNorm and utilized the properties of CausalConv3D to achieve lossless inference through temporal dimension chunking. In each layer of CausalConv3D, we cache the information from the previous few frames to maintain continuity during the convolution sliding operation for the next temporal chunk, thereby enabling lossless processing. As illustrated, we use a kernel size of 3 and a stride of 1 as an example:

**Initial Chunk (chunk idx=0):** For the first time chunk, we perform standard causal padding to support joint processing of graphs and videos. After the convolution operation, we cache the last two frames of this chunk into the causal cache in preparation for the next chunk's inference.

**Subsequent Chunks (chunk idx=1 and beyond):** Starting from the second time chunk, we no longer use causal padding. Instead, we concatenate the cached causal information from the previous chunk to the front of the current chunk. We continue to cache the last two frames of the current input into the causal cache for use in subsequent chunks.

## Prompt Refiner

User-provided captions are typically fewer than 10 words, whereas the text annotations in the current training data are often dense. This inconsistency between training and inference may result in poor visual quality and weak text alignment. We categorize captions into four types:

(1) Short captions from real user input; we collected 11k from [COCO](https://cocodataset.org/#home).

(2) Captions composed of multiple tags; we collected 5k from [DiffusionDB](https://github.com/poloclub/diffusiondb).

(3) Medium-length captions generated by large language models; 3k sourced from [JourneyDB](https://github.com/JourneyDB/JourneyDB).

(4) Ultra-long, surrealist captions, sourced from Sora/Vidu/Pika/Veo and approximately 0.5k generated by GPT.

We used ChatGPT to rewrite the above captions, with the following instructions provided to ChatGPT:

```
rewrite the sentence to contain subject description action, scene description. 
Optional: camera language, light and shadow, atmosphere and
conceive some additional actions to make the sentence more dynamic,
make sure it is a fluent sentence, not nonsense.
```

Finally, we performed LoRA fine-tuning using [LLaMa 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), completing the training in just 30 minutes with a single H100. We fine-tuned for only 1 epoch, using a batch size of 32 and a LoRA rank of 64. The log can be found [here](https://api.wandb.ai/links/1471742727-Huawei/p5xmkft5).

### Data Construction

We randomly sampled from the original Panda70m dataset and found many videos to be static, contain multiple subtitles, or suffer from motion blur. Additionally, the captions in Panda70m did not always accurately describe the video content. To address this, we designed a video filtering pipeline, which retained approximately 27% of the videos after processing.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/90f9d386-ff2e-465a-b013-a9e7151afaf8" height=400 />
</figure>
</center>


#### Jump Cut and Detect Motion

We used [LPIPS](https://github.com/richzhang/PerceptualSimilarity) frame-skipping to compute inter-frame semantic similarity, identifying anomalies as cut points and taking the mean as the motion score. We found that videos with motion scores below 0.001 were nearly static, while those above 0.3 exhibited significant jitter and flicker. After applying this method, we manually reviewed 2k videos and concluded that the cut detection accuracy was sufficient for pre-training requirements.

#### OCR

We estimated the average position of subtitles on common video platforms to be around 18%. Consequently, we set the maximum crop threshold to 20% of the video's original dimensions and used [EasyOCR](https://github.com/JaidedAI/EasyOCR) to detect subtitles (sampling one frame per second). However, not all videos have subtitles or printed text located at the edges; this method may miss text appearing in central areas, such as in advertisement videos or speeches. Nonetheless, we cannot assume that the presence of text in a video necessitates filtering it out, as certain texts in specific contexts can be meaningful. We leave such judgments to aesthetic considerations.

#### Aesthetic

As before, we used the [Laion aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) for evaluation. Based on the visualization [website](http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html), we determined that a score of 4.75 serves as a suitable threshold, effectively filtering out excessive text while retaining high-quality aesthetics. We will add an additional aesthetic prompt, such as `A high-aesthetic scene, ` for data with a score above 6.25.

#### Video Quality

Some old photos or videos have very low bit rates, resulting in blurry visual effects even at 480P resolution, often resembling a mosaic appearance. Aesthetic filtering struggles to exclude these videos, as it resizes images to 224 resolution. We aim to establish a metric for assessing absolute video quality, independent of the visual content itself, focusing solely on compression artifacts, low bit rates, and jitter. We employed the technical prediction score from [DOVER](https://github.com/VQAssessment/DOVER) and excluded videos with scores below 0.

#### Recheck Motion

Since some videos contain subtitles, variations in the subtitles may lead to inaccurate motion values. Therefore, we re-evaluated the motion values and discarded static videos.

#### Captioning

We used [QWen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) for video annotation.

```
Please describe the content of this video in as much detail as possible, 
including the objects, scenery, animals, characters, and camera movements within the video. 
Do not include '\n' in your response. 
Please start the description with the video content directly. 
Please describe the content of the video and the changes that occur, in chronological order.
```

However, the 7B model tends to generate certain prefixes, such as "This video" or "The video." We compiled a list of all irrelevant opening strings and removed them.

```
    'The video depicts ', 
    'The video captures ', 
    'In the video, ', 
    'The video showcases ', 
    'The video features ', 
    'The video is ', 
    'The video appears to be ', 
    'The video shows ', 
    'The video begins with ', 
    'The video displays ', 
    'The video begins in ', 
    'The video consists of ', 
    'The video opens with ', 
    'The video opens on ', 
    'The video appears to capture ', 
    'The video appears to show ', 
    "The video appears to depict ", 
    "The video opens in ", 
    "The video appears to focus closely on ", 
    "The video starts with ", 
    "The video begins inside ", 
    "The video presents ", 
    "The video takes place in ", 
    "The video appears to showcase ", 
    "The video appears to display ", 
    "The video appears to focus on ", 
    "The video appears to feature "
```

### Training Text-to-Video Diffusion Model

#### Framework

##### Skiparse (Skip-Sparse) Attention

In video generation models, alternating 2+1D spatial-temporal blocks is a commonly used approach, yet these models lack long-range modeling, limiting their performance ceiling. Consequently, models like  [CogVideoX](https://arxiv.org/abs/2408.06072), [Meta Movie Gen](https://ai.meta.com/research/movie-gen/), and Open-Sora Plan v1.2 employ **Full 3D Attention** as a denoiser, achieving substantially improved visual fidelity and motion quality compared to 2+1D models. This approach, however, requires calculating attention across all tokens in each clip encoding, which significantly raises training costs. For instance, Open-Sora Plan v1.2, training a 2.7-billion-parameter model, takes **100 seconds per step at 93x720p and over 15 seconds per step at 93x480p**, severely constraining scalability under limited computational resources.

To accelerate training while ensuring adequate performance, we propose the **Skiparse (Skip-Sparse) Attention** method. Specifically, under a fixed sparse ratio $$k$$ , we organize candidate tokens for attention through two alternating skip-gather methods. This approach preserves the attention operation is global while effectively reducing FLOPS, enabling faster training of 3D Attention models. In our experiments, applying Skiparse with sparse ratio $$k=4$$ to a 2.7B model reduced training time to **42 seconds per step at 93x720p and 8 seconds per step at 93x480p**.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/186377ca-26b2-4f0f-af42-ae6c846eebcb" />
</figure>
</center>

**Skiparse DiT modifies only the Attention component** within the Transformer Block, using two alternating Skip Sparse Transformer Blocks. With sparse ratio $$k$$, the sequence length in the attention operation reduces to $$\frac{1}{k}$$ of the original, and batch size increases by $$k$$-fold, lowering the theoretical complexity of self-attention to $$\frac{1}{k}$$ of the original, while cross-attention complexity remains unchanged. Due to GPU/NPU parallel processing, increasing the batch size by $$k$$-fold does not linearly decrease speed to $$\frac{1}{k}$$, resulting in a performance boost that exceeds theoretical expectations.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/80f9470a-8afe-4588-a22c-e8c576fea9b6" />
</figure>
</center>

In Single Skip mode, the elements located at positions $$[0, k, 2k, 3k, ...]$$ ,  $$[1, k+1, 2k+1, 3k+1, ...]$$ , ..., $$[k-1, 2k-1, 3k-1, ...]$$ are grouped into the same scope (with each list forming one scope of elements). The figure above, using $$k=2$$ as an example, illustrates this organizational structure. This concept is straightforward, as each token performs attention with tokens spaced $$k-1$$ apart.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/5880f667-7a06-4e7f-8e44-2e1cfb9209b8" />
</figure>
</center>

In Group Skip mode, elements at positions $$[(0, 1, ..., k-1), (k^2, k^2+1, ..., k^2+k-1), (2k^2, 2k^2+1, ..., 2k^2+k-1), ...]$$ , $$[(k, k+1, ..., 2k-1), (k^2+k, k^2+k+1, ..., k^2+2k-1), (2k^2+k, 2k^2+k+1, ..., 2k^2+2k-1), ...]$$ , ..., $$[(k^2-k, k^2-k-1, ..., k^2-1), (2k^2-k, 2k^2-k-1, ..., 2k^2-1), (3k^2-k, 3k^2 -k-1, ..., 3k^2-1), ...]$$ are grouped together as a scope (with each list forming a scope). This arrangement may seem complex numerically, so it can be helpful to understand with the above figure.

In this pattern, we first **group adjacent tokens** in segments of length $$k$$ , then **bundle these groups** with other groups that are spaced $$k-1$$ groups apart into a single scope.For example, in $$[(0, 1, ..., k-1), (k^2, k^2+1, ..., k^2+k-1), (2k^2, 2k^2+1, ..., 2k^2+k-1), ...]$$ , each set of indices in parentheses represents a group. Each group is then connected with another group that is offset by $$k-1$$ groups, forming one scope.

Since the last index of the first group is $$k-1$$ , the first token in the next group to be linked will be at index $$k-1+k(k-1)+1=k^2$$ . Following this pattern, you can determine the indices for each scope in this configuration.

##### Why "Skiparse"?

The 2+1D DiT models temporal understanding only along the time axis of a single spatial location, theoretically and practically limiting performance. In real-world scenarios, changes at a specific spatial location are typically influenced not by prior content at that same location but by content across all spatial locations at preceding times. This constraint makes it challenging for 2+1D DiT to model complex physical dynamics accurately.

Full 3D Attention represents global attention, allowing any spatial position at any time to access information from any other position across all times, aligning well with real-world physical modeling. However, this approach is time-consuming and inefficient, as visual information often contains considerable redundancy, making it unnecessary to establish attention across all spatiotemporal tokens.

**A ideal spatiotemporal modeling approach should employ attention that minimizes the overhead from redundant visual information while capturing the complexities of the dynamic physical world**. Reducing redundancy requires avoiding connections among all tokens, yet global spatiotemporal attention remains essential for modeling complex physical interactions.

To achieve a balance between 2+1D efficiency and Full 3D’s strong spatiotemporal modeling, we developed Skiparse Attention.  This approach provides global spatiotemporal attention within each block, with each block having the same “receptive field”. The use of "group" operations also introduces a degree of locality, aligning well with visual tasks.

Interestingly, once you understand the Skiparse Attention mechanism, you’ll notice that **the attention in 2+1D DiT corresponds to a sparse ratio of $$k=HW$$  (since $$T \ll HW$$ , making the "skip" in Group Skip negligible), while Full 3D DiT corresponds to a sparse ratio of $$k=1$$.** In Skiparse Attention, $$k$$ is typically chosen to be close to 1, yet far smaller than $$HW$$ , making it a 3D Attention that approaches the effectiveness of Full 3D Attention.

In Skiparse Attention, Single Skip is a straightforward operation, easily understood by most. Within Group Skip, the Group operation is also intuitive, serving as a means to model local information. However, **Group Skip involves not only grouping but also skipping**—particularly between groups—which is often overlooked. This oversight frequently leads researchers to confuse Skiparse Attention with a Skip + Window Attention approach. The key difference lies in even-numbered blocks: Window Attention only groups tokens without skipping between groups. The distinctions among these attention methods are illustrated in the figure below, which shows the attention scopes for self-attention only, with dark tokens representing the tokens involved in each attention calculation.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/62d0c75a-7e1d-458e-9faf-ae394e8ddd34" />
</figure>
</center>

To deeply understand why nearly global attention is necessary and why Skiparse Attention theoretically approximates Full 3D Attention more closely than other common methods, we introduce the concept of **Average Attention Distance**. This concept is defined as follows: for any two tokens, if it takes $$m$$ attention operations to establish a connection between them, the attention distance is  $$m$$ . The average attention distance for a tensor is then the mean of the attention distances across all token pairs, representing the corresponding attention method’s overall connectivity efficiency. The average attention distance of all tokens within a tensor is defined as the average attention distance for that particular attention method. 

For example, in Full 3D Attention, any token can connect with any other token in just one attention operation, resulting in an average attention distance of 1.

In 2+1D Attention, the process is somewhat more complex, though still straightforward to understand. In all configurations above, any two different tokens can connect with an attention distance between 1 and 2 (Note that we define the attention distance between a token and itself as zero). Thus, for the other three attention methods, we can first identify which tokens have an attention distance of 1. Subsequently, tokens with an attention distance of 2 can be determined, allowing us to calculate the average attention distance.

In the $$2N$$ Block, attention operates over the $$(H, W)$$ dimensions, where tokens within this region have an attention distance of 1. In the $$2N+1$$ Block, attention operates along the $$(T)$$ dimension, also assigning an attention distance of 1 for these tokens. The total number of tokens with an attention distance of 1 in this case is $$HW + T - 2$$ (excluding the token itself, hence $$(HW + T - 1) - 1 = HW + T - 2$$).

Therefore, in 2+1D Attention, the average attention distance (AVG Attention Distance) is:

$$
\begin{aligned}
	d&=\frac{1}{THW}\left[ 1\times 0+\left( HW+T-2 \right) \times 1+\left[ THW-\left( HW+T-1 \right) \right] \times 2 \right]\\
	&=2-\left( \frac{1}{T}+\frac{1}{HW} \right)\\
\end{aligned}
$$

In Skip+Window Attention, aside from the token itself, there are $$\frac{THW}{k} - 1$$ tokens with an attention distance of 1 in the $$2N$$ Block, and $$k - 1$$ tokens with an attention distance of 1 in the $$2N+1$$ Block. Thus, the total number of tokens with an attention distance of 1 is $$\frac{THW}{k} + k - 2$$.

Therefore, in Skip+Window Attention, the average attention distance (AVG Attention Distance) is:

$$
\begin{aligned}
	d&=\frac{1}{THW}\left[ 1\times 0+\left( \frac{THW}{k}+k-2 \right) \times 1+\left[ THW-\left( \frac{THW}{k}+k-1 \right) \right] \times 2 \right]\\
	&=2-\left( \frac{1}{k}+\frac{k}{THW} \right)\\
\end{aligned}
$$

In Skiparse Attention, aside from the token itself, $$\frac{THW}{k} - 1$$ tokens have an attention distance of 1 in the $$2N$$ Block, and $$\frac{THW}{k} - 1$$ tokens have an attention distance of 1 in the $$2N+1$$ Block. Notably, $$\frac{THW}{k^2} - 1$$ tokens can establish an attention distance of 1 in both blocks and should not be counted twice.

Therefore, in Skiparse Attention, the average attention distance (AVG Attention Distance) is:

$$
\begin{aligned}
	d&=\frac{1}{THW}\left[ 1\times 0+\left[ \frac{2THW}{k}-2-\left( \frac{THW}{k^2}-1 \right) \right] \times 1+\left[ THW-\left( \frac{2THW}{k}-\frac{THW}{k^2} \right) \right] \times 2 \right]\\
	&=2-\frac{2}{k}+\frac{1}{k^2}-\frac{1}{THW}\\
	&=2-\frac{2}{k}+\frac{1}{k^2}\left( 1\ll THW \right)\\
\end{aligned}
$$

In fact, in the Group Skip of the $$2N+1$$ Block, the actual sequence length is $$k\lceil \frac{THW}{k^2} \rceil$$ rather than $$\frac{THW}{k}$$. The prior calculation assumes the ideal case where $$k \ll THW$$ and $$k$$ divides $$THW$$ exactly, yielding $$k\lceil \frac{THW}{k^2} \rceil = k \cdot \frac{THW}{k^2} = \frac{THW}{k}$$. In practical applications, excessively large $$k$$ values are typically avoided, making this derivation a reasonably accurate approximation for general use.

Specifically, when $$k = HW$$ and padding is disregarded, since $$T \ll HW$$, group skip attention reduces to window attention with a window size of $$HW$$. Given that padding does not affect the final computation, Skiparse Attention is equivalent to 2+1D Attention when $$k = HW$$.

For the commonly used resolution of 93x512x512, using a causal VAE with a 4x8x8 compression rate and a DiT with a 1x2x2 patch embedding, we obtain a latent shape of 24x32x32 before applying attention. The AVG Attention Distance for different calculation methods would then be as follows:

|                        | Full 3D Attention | 2+1D  Attention |
| ---------------------- | ----------------- | --------------- |
| AVG Attention Distance | 1                 | 1.957           |

|                        | Skip + Window Attention(k=2) | Skip + Window Attention(k=4) | Skip + Window Attention(k=6) | Skip + Window Attention(k=8) |
| ---------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| AVG Attention Distance | 1.500                        | 1.750                        | 1.833                        | 1.875                        |

|                        | Skiparse Attention(k=2) | Skiparse Attention(k=4) | Skiparse Attention(k=6) | Skiparse Attention(k=8) |
| ---------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| AVG Attention Distance | 1.250                   | 1.563                   | 1.694                   | 1.766                   |

In 2+1D Attention, the average attention distance is 1.957, larger than that of Skip + Window Attention and Skiparse Attention at commonly used sparse ratios. While Skip + Window Attention achieves a shorter average attention distance, its modeling capacity remains limited due to the locality of attention in its 2N+1 blocks. Skiparse Attention, with the shortest average attention distance, applies global attention in both 2N and 2N+1 blocks, making its spatiotemporal modeling capabilities closer to Full 3D Attention than the other two non-Full 3D methods.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/80ca6d70-5033-454b-883f-11d12d140360" width=600/>
</figure>
</center>

The figure above shows how Skiparse Attention’s AVG Attention Distance changes with sparse ratio $$k$$.

We can summarize the characteristics of these attention types as follows:

|                                    | Full 3D Attention | 2+1D  Attention                  | Skip + Window Attention                 | Skiparse Attention                                           |
| ---------------------------------- | ----------------- | -------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| Speed                              | Slow              | Fast                             | Depending on $$k$$                    | Depending on $$k$$                                         |
| Spatiotemporal modeling capability | Strong            | Weak                             | Weak                                    | Approaches Full 3D                                           |
| Is attention global?               | Yes               | No                               | Half of the attention blocks are global | Yes                                                          |
| Computation load per block         | Equal             | Not Equal                        | Not Equal                               | Equal                                                        |
| AVG Attention Distance             | 1                 | $$2-(\frac{1}{T}+\frac{1}{HW})$$ | $$2-(\frac{1}{k}+\frac{k}{THW})$$       | $$2-\frac{2}{k}+\frac{1}{k^2},1<k\ll THW$$                   |

Considering both computational load and AVG Attention Distance, we select Skiparse with $$k = 4$$, replacing the first and last two blocks with Full 3D Attention to enhance performance.

Overall, we retained the architecture from version 1.2 but incorporated Skiparse Attention module.


<center>
<figure>
	<img src="https://github.com/user-attachments/assets/4525468f-78be-4b03-9609-9511bded95b1" height=350 />
</figure>
</center>

#### Dynamic training

Overall, we maintained the bucket strategy from v1.2.0, pre-defining the shape of each video during training and aggregating data of the same shape through a sampler. Finally, the dataloader retrieves data based on our aggregated indices.

In our early implementation, we specified `--max_width`, `--max_height`, `--min_width`, and `--min_height`. While this allows for specifying arbitrary resolutions within a certain range, this approach can easily lead to OOM issues during video training. For instance, for a 720P (720×1280) video, if the maximum dimensions are set to 720, the video would be scaled to 405×720. However, if there are square videos with resolutions greater than 720, they would be scaled to 720×720. Most videos are non-square, and to prevent OOM, we need to reserve GPU memory, which leads to significant computational waste. Therefore, we recommend using `--max_token` and `--min_token` to limit any range, as this aligns better with the Transformer architecture.

#### Training scheduler

We replaced the eps-pred loss with v-pred loss and enable ZeroSNR. For videos, we resample to 16 FPS for training.

**Stage 1**: We initially initialized from the image weights of version 1.2.0 and trained images at a resolution of 1x320x320. The objective of this phase was to fine-tune the 3D dense attention model to a sparse attention model. The entire fine-tuning process involved approximately 100k steps, with a batch size of 1024 and a learning rate of 2e-5. The image data was primarily sourced from SAM in version 1.2.0.


**Stage 2**: We trained the model jointly on images and videos, with a maximum resolution of 93x320x320. The entire fine-tuning process involved approximately 300k steps, with a batch size of 1024 and a learning rate of 2e-5. The image data was primarily sourced from SAM in version 1.2.0, while the video data consisted of the unfiltered Panda70m. In fact, the model had nearly converged around 100k steps, and by 300k steps, there were no significant gains. Subsequently, we performed data cleaning and caption rewriting, with further data analysis discussed at the end.

**Stage 3**: We fine-tuned the model using our filtered Panda70m dataset, with a fixed resolution of 93x352x640. The entire fine-tuning process involved approximately 30k steps, with a batch size of 1024 and a learning rate of 1e-5.

### Training Image-to-Video Diffusion Model

#### Framework

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/41e22292-8d8b-469e-940a-6e5ae00bf620" />
</figure>
</center>

In terms of framework, Open-Sora Plan v1.3 continues to use the Inpainting model architecture from Open-Sora Plan v1.2.

#### Data processing

For data processing, Open-Sora Plan v1.3 introduces two new mask types: an all-1 mask and an all-0 mask. This brings the total number of mask types in the Inpainting Model to six.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/f31b222e-811c-49b9-839c-f72fb85c4ee4" />
</figure>
</center>

In the figure above, black indicates retained frames, while white denotes discarded frames. The corresponding frame strategies are as follows:

- **Clear**: Retain all frames.
- **T2V**: Discard all frames.
- **I2V**: Retain only the first frame; discard the rest.
- **Transition**: Retain only the first and last frames; discard the rest.
- **Continuation**: Retain the first $$n$$ frames; discard the rest.
- **Random**: Retain $$n$$ randomly selected frames; discard the rest.

#### Progressive training

The Open-Sora Plan v1.3 uses more data for training and employs a progressive training approach to help the model understand frame-based inpainting tasks.

Since the Inpainting Model supports various mask inputs, different mask inputs correspond to tasks of varying difficulty levels. Therefore, we can first teach the model simple tasks, such as random masks, allowing it to develop a basic capability for frame-based inpainting before gradually increasing the proportion of more challenging tasks. It is important to note that at different training stages, we ensure that at least 5% of the data the model sees pertains to T2V tasks, which is aimed at enhancing the model's understanding of prompts.

The model weights are initialized from the T2V model with zero initialization. The batch size is fixed at 256, and the learning rate is set to 1e-5, using a two-stage training approach.

**Stage 1**: Any resolution and duration within 93x102400 (320x320), using unfiltered motion and aesthetic low-quality data:

(1) Step 1: t2v 10%, continuation 40%, random mask 40%, clear 10%. Ensure that at least 50% of the frames are retained during continuation and random mask, training with 4 million samples.

(2) Step 2: t2v 10%, continuation 40%, random mask 40%, clear 10%. Ensure that at least 25% of the frames are retained during continuation and random mask, training with 4 million samples.

(3) Step 3: t2v 10%, continuation 40%, random mask 40%, clear 10%. Ensure that at least 12.5% of the frames are retained during continuation and random mask, training with 4 million samples.

(4) Step 4: t2v 10%, continuation 25%, random mask 60%, clear 5%. Ensure that at least 12.5% of the frames are retained during continuation and random mask, training with 4 million samples.

(5) Step 5: t2v 10%, continuation 25%, random mask 60%, clear 5%, training with 8 million samples.

(6) Step 6: t2v 10%, continuation 10%, random mask 20%, i2v 40%, transition 20%, training with 16 million samples.

(7) Step 7: t2v 5%, continuation 5%, random mask 10%, i2v 40%, transition 40%, training with 10 million samples.

**Stage 2:** Any resolution and duration within 93x236544 (e.g., 480x480, 640x352, 352x640), using filtered motion and aesthetic high-quality data:

t2v 5%, continuation 5%, random mask 10%, i2v 40%, transition 40%, training with 15 million samples.

#### About the Semantic Adapter

We conducted further experiments on the Semantic Adapter module and compared the video quality of Image-to-Video under various Image Encoders, including [Clip](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) and [Dino v2](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m). We also attempted strategies such as directly injecting image embeddings into cross-attention or extracting features from the Image Encoder using Qformer before injecting them into cross-attention. 

Under various strategies, we did not observe significant performance improvements; the impact on video quality was much smaller than that of the dataset. Therefore, we decided not to include the Semantic Adapter in Open-Sora Plan v1.3.

#### Noise Injection Strategy for Conditional Images

Researchs like [CogVideoX](https://arxiv.org/abs/2408.06072) and [Stable Video Diffusion](https://stability.ai/stable-video) have indicated that adding a certain amount of noise to Conditional Images can enhance the generalization capability of I2V models and achieve a greater range of motion. Therefore, we will implement this strategy in Open-Sora Plan v1.3, and a model utilizing this approach will be released shortly.

### The implementation of Skiparse Attention

Skiparse is theoretically easy to understand and straightforward to implement. Its implementation mainly relies on the rearrange operation, which reduces the sequence length of latents before entering `F.scaled_dot_product_attention()`. Aside from this adjustment, no other modifications are made. For simplicity, the following discussion focuses solely on the self-attention part, excluding the attention mask.

The pseudocode implementation of Single Skip is as follows:

```python
# x.shape: (B,N,C)
def single_skip_rearrange(x, sparse_k):
	return rearrange(x, 'b (g k) d -> (k b) g d', k=sparse_k)
def reverse_sparse(x, sparse_k):
	return rearrange(x, '(k b) g d -> b (g k) d', k=sparse_k)
q, k, v = Q(x), K(x), V(x)
q = add_rope(q)
k = add_rope(k)
q = single_skip_rearrange(q)
k = single_skip_rearrange(k)
v = single_skip_rearrange(v)
hidden_states = F.scaled_dot_product_attention(q=q,k=k,v=v)
output = reverse_sparse(hidden_states)
```

The core of the Skiparse operation lies in "rearranging the sequence", which corresponds to the Single Skip operation in the pseudocode:

```python
rearrange(x, '(g k) b d -> g (k b) d', k=sparse_k)
```

This operation can be understood as a combination of a reshape and a transpose operation:

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/e42c4dd5-ee95-42a8-b8c6-8cb803cd7e12" height=300/>
</figure>
</center>

In this way, $$k$$ sub-sequences can be created, and $$k$$ can be moved to the batch dimension, allowing the Attention mechanism to compute the sub-sequences in parallel.

Understanding Single Skip makes Group Skip easy to comprehend as well; it simply adds a grouping operation before the Skip. Its pseudocode is as follows:

```python
# x.shape: (B,N,C)
def group_skip_rearrange(x, sparse_k):
	return rearrange(x, ' b (n m k) d -> (m b) (n k) d', m=sparse_k, k=sparse_k)
def reverse_sparse(x, sparse_k):
	return rearrange(x, '(m b) (n k) d -> b (n m k) d', m=sparse_k, k=sparse_k)
q, k, v = Q(x), K(x), V(x)
q = add_rope(q)
k = add_rope(k)
q = group_skip_rearrange(q)
k = group_skip_rearrange(k)
v = group_skip_rearrange(v)
hidden_states = F.scaled_dot_product_attention(q=q,k=k,v=v)
output = reverse_sparse(hidden_states)
```

Every $$k^2$$ tokens form a repetition, and every $$k$$ tokens form a group. To help everyone better understand this operation, the following figure illustrates the situation when $$k=3$$:

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/5e777862-d03c-4c7e-8ffc-1e1e9234b84e"/>
</figure>
</center>

It is important to note that the rope is added before the Skiparse operation and cannot be placed after it, as the sequence after Skiparse will lose its original spatial positions.


## Future Work and Discussion

### CasualVideoVAE
For videos, increasing the compression ratio while maintaining the original latent dimension leads to significant information loss. Therefore, it is a trend to increase the latent dimension to achieve higher compression ratios. A more advanced VAE will be released in the next version.

### Diffusion Model
The current 2B model in version 1.3.0 shows performance saturation during the later stages of training. However, it does not perform well in understanding physical laws (e.g., a cup overflowing with milk, a car moving forward, or a person walking). We have 4 hypotheses regarding this issue:

#### The current data domain is too narrow.

We randomly sampled 2,000 videos from Panda70m and conducted manual verification, finding that less than 1% featured cars in motion, and there were even fewer than 10 videos of people walking. Approximately 80% of the videos consist of half-body conversations with multiple people in front of the camera. Therefore, we speculate that the narrow data domain of Panda70m restricts the model's ability to generate many scenarios. We plan to collect more data in the next version.

#### Joint training of images and videos

Models such as [Open-Sora v1.2](https://github.com/hpcaitech/Open-Sora), [EasyAnimate v4](https://github.com/aigc-apps/EasyAnimate), and [Vchitect-2.0](https://github.com/Vchitect/Vchitect-2.0) can easily generate high-visual-quality videos, possibly due to their direct inheritance of image weights ([Pixart-Sigma](https://pixart-alpha.github.io/PixArt-sigma-project/), [HunyuanDiT](https://github.com/Tencent/HunyuanDiT), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)). They train the model with a small amount of video data to learn how to flow along the time axis based on 2D images. However, we trained images from scratch with only 10M-level data, which is far from sufficient. We have two hypotheses regarding the training strategy: (1) the first is to start joint training from scratch, with images significantly outnumbering videos; (2) The second is to first train a high-quality image model and then use joint training, with a higher proportion of videos at that stage. Considering the learning path and training costs, the second approach may offer more decoupling, while the first aligns better with scaling laws.

#### The model still needs to scale

By observing the differences between [CogVideoX-2B](https://github.com/THUDM/CogVideo) and its 5B variant, we can clearly see that the 5B model understands more physical laws than the 2B model. We speculate that instead of spending excessive effort designing for smaller models, it may be more effective to leverage scaling laws to solve these issues. In the next version, we will scale up the model to explore the boundaries of video generation.

We currently have two plans: one is to continue using the Deepspeed/FSDP approach, sharding the EMA and text encoder across ranks with Zero3, which is sufficient for training 10-15B models. The other is to adopt [MindSpeed](https://gitee.com/ascend/MindSpeed) for various parallel strategies, enabling us to scale the model up to 30B.

#### Supervised loss in training

Whether flow-based models are more suitable than v-pred models remains uncertain and requires further ablation studies to determine.

### How else can "Skiparse" skip?

The sparse method we use is theoretically and practically straightforward; however, its implementation treats the original video data purely as a one-dimensional sequence, neglecting the 2D spatial priors. Thus, we extended Skiparse to create Skiparse-2D, which is better suited for 2D Visuals.

<center>
<figure>
	<img src="https://github.com/user-attachments/assets/44bd5284-b4c0-4a9d-9f2e-5acbb2e3450f" height=500/>
</figure>
</center>

In Skiparse-2D, a sparse ratio of $$k$$ represents the sparsity along the $$h$$ or $$w$$ direction. In terms of the number of tokens involved in attention computation, it is equivalent to the square of the sparse ratio in Skiparse-1D.

We conducted basic experiments comparing Skiparse-1D and Skiparse-2D. Under identical experimental settings, Skiparse-2D showed no improvement over Skiparse-1D in terms of loss or sampling results. Additionally, Skiparse-2D is less flexible to implement than Skiparse-1D. Therefore, we opted to use the Skiparse-1D approach for training in Open-Sora Plan v1.3.

Nevertheless, given our limited experimentation, the feasibility of Skiparse-2D remains worth exploring. Intuitively, Skiparse-2D better aligns with the spatial characteristics of visuals, and as sparse ratio $$k$$ increases, its approach intuitively approximates that of 2+1D. We therefore encourage interested researchers in the community to pursue further exploration in this area.