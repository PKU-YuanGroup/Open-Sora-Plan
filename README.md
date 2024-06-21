# Variable-Resolution T2V Training with NaViT
## Introduction
In the [technical characteristics](https://openai.com/index/sora/) disclosed by SORA officials, SORA is trained using a Transformer-based model on visual data of varying durations, resolutions, and aspect ratios , which may be crucial for reproducing the effects of SORA.
> We represent videos and images as collections of smaller units of data called patches, each of which is akin to a token in GPT. By unifying how we represent data, we can train diffusion transformers on a wider range of visual data than was possible before, spanning different durations, resolutions and aspect ratios.


However, video data are usually of different resolutions, making it impossible to concatenate a number of training data into a batched sequence for parallel computation. The community has proposed the following solutions:

1. Padding: Padding all videos in a batch to the length of the longest sequence, similar to the techniques commonly used in NLP. However, this introduces a lot of unnecessary computation, especially when there is a high variance in the resolution of training samples.
2. Bucketing: Dividing training samples into different "buckets" based on their resolution, then randomly drawing samples from the same bucket to compose a batch. This, however, disrupts the sampling mechanism of SGD and may affect model convergence.
3. NaViT: NaViT was proposed for image classification tasks to train with images of different resolutions. By packing short sequences of different lengths into one long sequence and using an attention mask to isolate the attention computation for different samples, NaViT significantly reduces the proportion of unnecessary padding while ensuring computational equivalence, thereby improving the efficiency of variable-resolution training. However, in text-to-video (t2v) tasks, due to the introduction of text conditions and temporal information, integrating NaViT becomes more complex. As of now, there is no precedent in the industry for implementing NaViT for video generation tasks.

Therefore, we achieve a NaViT implementation specifically for Open-SoRA-Plan, hoping to make an effort to AIGC community.
## How to Use
1. Prepare datasets and install requirements for training following the original Open-Sora-Plan procedure.
2. Start your navit training
```
sh scripts/text_condition/train_videoae_65_navit.sh
```
3. You might want to test its numerical consistency with the original Open-Sora-Plan implementation.
```
sh scripts/text_condition/train_videoae_65_navit_test.sh
```
## Method
### Dataloader
In the preprocessing stage, we introduce RandomResize transform that randomly resizes inputs to a range between 64 and the original resolution, to simulate variable resolution training. This transform should be removed during training so that the videos are already in their original resolutions.

Concurrently, we upscales the video to the nearest integer multiple of the product of the VAE's compression ratio and patch size. This adjustment guarantees that the resultant video latent representation is divisible by the patch size. For image-joint-training, we further resize the images to their corresponding video resolution.

Furthermore, modifications to the DataLoader have been implemented to yield the list of Tensors rather than batching them together to support the samples with different resolutions.
### VAE encode
Due to the inherent limitations of the existing variational autoencoder (VAE) architecture, which does not accommodate serialized data formats, the VAE encoding step is necessarily positioned preceding the data packing process. Moreover, the VAE lacks the capability to concurrently handle inputs of different resolutions. Consequently, we loop over the individual video samples and use VAE to encode them sequentially.
### Video Packing
In Latte, patch embedding is achieved through a 2D convolutional layer (conv2d) followed by reshaping operations to yield a serialized output sequence. We also follow this procedure, except that we group and pack some videos with different resolutions into a single sequence.

Currently, a simple grouping strategy is used where videos are sequentially grouped based on a maximum sequence length threshold. Specifically, when the combined length of grouped videos exceeds this threshold, a new group is started. Videos within the same group are concatenated into a single sequence.

Each video within a group undergoes a shared-parameter conv2d operation followed by reshaping to a serialized format. The resulting sequences within the same group are then concatenated along the token dimension, effectively completing the video packing process.

To facilitate distinction between tokens from different samples as well as the padding tokens during subsequent attention mechanisms, a token-wise labeling scheme is implemented. Here, unique identifiers (0, 1, 2, ...) denote tokens belonging to different videos, while padding tokens are consistently marked with -1.
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/jpeg/124356420/1716884849686-cc58b714-86e3-4da8-a318-541c410e45d9.jpeg)
### Token drop
We incorporate token dropping proposed in NaViT, enabling us to control the sequence length after the packing process. This strategy involves selectively discarding certain tokens to manage the overall sequence length, which is particularly beneficial for managing computational complexity and potentially mitigating the effects of less informative tokens.
### Position Encoding packing
For videos of arbitrary resolutions and aspect ratios, we interpolate their positional encodings to a fixed resolution, which can be specified by the user. This standardization facilitates the utilization of pre-trained models that were trained at a specific resolution.

Within the same group of videos, after interpolating their positional encodings to match the designated resolution, they are concatenated along the token dimension as well.
### Timestep embedding packing
Within a single batch, since videos have their individual timesteps during training, we pack these timesteps according to the videos' packing pattern. This approach ensures that timestep embeddings can effectively integrate into the attention operations.
### Text condition packing
Similarly to packing timesteps, text inputs are also packed in accordance with the video's established packing pattern. This method ensures the alignment  between textual and visual information.

Akin to the video token labeling, text tokens are also annotated. Each token is systematically tagged with identifiers (0, 1, 2, ...), distinguishing text originating from different videos. Consistently, padding tokens within the text sequences are labeled with -1. This token-level differentiation is crucial for maintaining the consistency of multi-sample processing in downstream attention-based mechanisms.

Additionally, text data inherently includes a mask indicating positions where padding was introduced during tokenization. These tokens are also labeled with -1. For simplicity, we omit this in the illustration.
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/jpeg/124356420/1718863469086-00c57918-7f8c-45ce-9c8c-ac3a56a040c2.jpeg)
### Attention mask
#### self-attention mask
The query, key, and value (Q, K, V) are all derived from the respective video in self-attention. Our attention mechanism is configured to compute exclusively the tokens belonging to the same video, while effectively filtering out padding tokens. We utilize pre-saved token labels to generate the mask.
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/jpeg/124356420/1716894781927-fb25ea3f-93b6-42c0-b565-364e6ba5364c.jpeg)
#### cross-attention mask
In cross-attention, The query (Q) is generated from the video, while the key and value (K, V) are derived from the text. Our cross-attention mechanism is designed to compute attention only for the text and video tokens in the same video, excluding any padding tokens. To generate the cross-attention mask, we concurrently utilize the pre-saved labels for both video tokens and text tokens.
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/jpeg/124356420/1716894782010-82000b1d-9079-4e36-ac11-9302934524ca.jpeg)
### Loss
The predicted noise from the model needs to be compared with the actual noise to compute the loss. One approach involves unpacking the noise sequence the model outputs back into the original video format according to the saved packing pattern, and then computing the loss with respect to the ground truth noise. However, due to the variable resolution of the original videos, it is not feasible to batch process them for loss computation in a parallel manner.

Therefore, we alternatively pack the ground truth noise using the same packing pattern and compute the loss with the model's output sequence. Since the packed sequences have a fixed length, this allows for parallelized acceleration of the loss computation.
## Numerical Consistency
We tested the consistency between the outputs of the NaViT model and the original model on the raw dataset provided by Open-Sora-Plan. Both models takes the same input data (with a fixed resolution of 512x512) and identical Position Encoding parameters. The output of the original model was packed according to the same pattern and compared with the output sequences of NaViT. The test script is attached in the code for reproduction.
## Limitation
1.  Currently, we use a naive sequential grouping strategy rather than an optimized greedy grouping strategy.
2. Due to the significant workload involved in adapting the KL loss for NaViT, only MSE loss is currently supported.
3. Packing has only been achieved for the spatial dimension, and does not support the input with different durations.
4. Since NaViT requires attention masks to achieve computational isolation between different videos in a sequence, flash attention cannot be used for acceleration.
