## Report v1.5.0

在2024年的10月，我们发布了Open-Sora Plan v1.3.0，第一次将一种稀疏化的attention结构——skiparse attention引入video generation领域。同时，我们采用了高效的WFVAE，使得训练时的编码时间和显存占用大大降低。

在Open-Sora Plan v1.5.0中，Open-Sora Plan引入了几个关键的更新：

1、更好的sparse dit——SUV。在skiparse attention的基础上，我们将sparse dit扩展至U形变化的稀疏结构，使得在保持速度优势的基础上sparse dit可以取得和dense dit相近的性能。

2、更高压缩率的WFVAE。在Open-Sora Plan v1.5.0中，我们尝试了8x8x8下采样率的WFVAE，它在性能上媲美社区中广泛存在的4x8x8下采样率的VAE的同时latent shape减半，降低attention序列长度。

3、data和model scaling。在Open-Sora Plan v1.5.0中，我们收集了1.1B的高质量图片数据和40m的高质量视频数据，并将模型大小scale到8.5B，使最终得到的模型呈现出不俗的性能。

4、更简易的Adaptive Grad Clipping。相比于version 1.3.0中较复杂的丢弃污点batch的策略，在version 1.5.0中我们简单地维护一个adaptive的grad norm threshold并clipping，以此更适应各种并行策略的需要。

Open-Sora Plan v.1.5.0全程在昇腾910系列加速卡上完成训练和推理，并采用mindspeed-mm训练框架适配并行策略。

### Open-Source Release

Open-Sora Plan v1.5.0的开源包括：

1、所有训练和推理代码。你也可以在[MindSpeed-MM](https://gitee.com/ascend/MindSpeed-MM)官方仓库找到open-sora plan v1.5.0版本的实现。

2、8x8x8下采样的WFVAE权重以及8.5B的SUV去噪器权重。

## Detailed Technical Report

### Data collection and processing

我们共收集了来自Recap-DataComp-1B、Coyo700M、Laion-aesthetic的共1.1B图片数据。对于图片数据，我们不进行除了分辨率之外的筛选。我们的视频数据来自于Panda70M以及其他自有数据。对于视频数据，我们采用与Open-Sora Plan v1.3.0相同的处理策略进行筛选，最终数据量为40m的高质量视频数据。

### Adaptive Grad Clipping

在Open-Sora Plan v1.3.0中，我们介绍了一种基于丢弃梯度异常batch的Adaptive Grad Clipping策略，这种策略具有很高的稳定性，但是执行逻辑过于复杂。因此，在Open-Sora Plan v1.5.0中，我们选择将该策略进行优化，采用EMA方式维护grad norm的threshold，并在grad norm超过该threshold时裁剪到threshold以下。该策略本质上是将大模型领域常用的1.0常数grad norm threshold扩展为一个随着训练进程动态变化的threshold。

```python
'''
	moving_avg_max_grad_norm: EMA方式维护的最大grad norm
	moving_avg_max_grad_norm_var: EMA方式维护的最大grad norm的方差
	clip_threshold: 根据3 sigma策略计算得到的梯度裁剪阈值
	ema_decay: EMA衰减系数，一般为0.99
	grad_norm: 当前step的grad norm
'''
clip_threshold = moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)
if grad_norm <= clip_threshold:
    # grad norm小于裁剪阈值，则该step参数正常更新，同时更新维护的moving_avg_max_grad_norm 和 moving_avg_max_grad_norm_var
    moving_avg_max_grad_norm = ema_decay * moving_avg_max_grad_norm + (1 - ema_decay) * grad_norm
    max_grad_norm_var = (moving_avg_max_grad_norm - grad_norm) ** 2
    moving_avg_max_grad_norm_var = ema_decay * moving_avg_max_grad_norm_var + (1 - ema_decay) * max_grad_norm_var
    参数更新...
else:
    # grad norm大于裁剪阈值，则先裁剪grad使grad norm减少至clip_threshold，再进行参数更新。
    clip_coef = grad_norm / clip_threshold
    grads = clip(grads, clip_coef) # 裁剪grads
    参数更新...
```

该策略相较于v1.3.0中策略实现更简单，且能够很好应对diffusion训练后期grad norm远小于1.0时仍存在loss spike的问题。

### WFVAE with 8x8x8 downsampling

在V1.5.0版本中，我们将VAE的时间压缩率从4倍压缩提高至8倍压缩，使得对于同样原始尺寸的视频，latent shape减少为先前版本的一半，这使得我们可以实现更高帧数的视频生成。

| Model             | THW(C)        | PSNR         | LPIPS         | rFVD         |
| ----------------- | ------------- | ------------ | ------------- | ------------ |
| CogVideoX         | 4x8x8 (16)    | <u>36.38</u> | 0.0243        | <u>50.33</u> |
| StepVideo         | 8x16x16 (16)  | 33.61        | 0.0337        | 113.68       |
| LTXVideo          | 8x32x32 (128) | 33.84        | 0.0380        | 150.87       |
| Wan2.1            | 4x8x8 (16)    | 35.77        | **0.0197**    | **46.05**    |
| Ours （WF-VAE-M） | 8x8x8 (32)    | **36.91**    | <u>0.0205</u> | 52.53        |

**Test on an open-domain dataset with 1K samples.**

WFVAE详情请见[WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model](https://arxiv.org/abs/2411.17459)

### Training Text-to-Video Diffusion Model

#### Framework —— SUV: A Sparse U-shaped Diffusion Transformer For Fast Video Generation

在Open-Sora Plan v1.3.0中，我们讨论了Full 3D Attention以及2+1D Attention的优劣，并综合他们的特点提出了Skiparse Attention——一种新型的global sparse attention，具体原理请参考[Report-v1.3.0](https://github.com/yunyangge/Open-Sora-Plan/blob/main/docs/Report-v1.3.0.md)。

在一个事先指定的sparse ratio $k$ 下，Skiparse Attention按照Single Skip - Group Skip交替的方式选定原序列长度 $\frac{1}{k}$ 的子序列进行attention交互，以此达到近似Full 3D Attention的效果。在Skiparse Attention中，sparse ratio越大，子序列在原序列中的位置越稀疏；sparse ratio越小，子序列在原序列中的位置越密集。但无论sparse ratio为多少，Skiparse Attention总是global的。

在Open-Sora Plan v1.5.0中，我们将这种稀疏交互方式看作一种token上的信息下采样，越稀疏的Skiparse Attention是一种更偏语义级的信息交互，越密集的Skiparse Attention是一种更偏细粒度的信息交互。遵循神经网络中多尺度设计的准则，我们在网络中引入U形变化稀疏度的Skiparse Attention，即浅层采用稀疏度低的Skiparse Attention，并在最浅层使用Full 3D Attention，深层采用稀疏度高的Skiparse Attention。特别的，类比UNet的设计，我们在相同稀疏度的Stage之间引入了Long Skip Connection。我们将这种U形变化的基于Skiparse Attention的DiT称之为SUV。

![SUV](https://img.picui.cn/free/2025/06/05/684108197cbb8.png)

在Open-Sora Plan v1.5.0中我们采用了基于MMDiT的SUV架构。对于video latents，我们对其进行skiparse attention操作，对于text embedding，我们仅对其进行repeat以对齐skiparse后的latent shape而不进行任何稀疏化操作。

SUV架构存在以下优点：

1、SUV是首个在视频生成模型上验证有效的稀疏化方法，在我们的消融实验中表明其在同样训练步数下可以达到接近dense dit的性能，且可以同时应用于预训练和推理中。

2、相较于UNet结构对feature map进行显式的下采样造成了信息损失，SUV的U形结构作用在Attention上，feature map的shape并没有发生变化，即信息并未发生损失，改变的只是token间信息交互的粒度。

3、Skiparse Attention及SUV不改变权重大小，只改变forward时attention的计算方式。这使得我们可以随着训练进程动态调整稀疏度，在图片训练或低分辨率视频训练时采用较低的稀疏度，在高分辨率视频训练时提高稀疏度，获得随序列长度近似线性增长的FLOPS。

对SUV架构更细致的分析，将会在后续更新至arxiv。

#### Training Stage

我们的训练包括Text-to-Image和Text-to-Video两个阶段。

#### Text-to-Image

先前的工作表明从合成数据训练得到的图像权重可能会影响视频训练时的效果。因此，在v1.5.0更新中，我们选择在更大的真实数据域内训练图像权重。我们收集了共1.1B的图片数据进行训练。由于图片存在多种不同的分辨率，而视频主要为9：16分辨率，因此我们选择在训练图片权重时开启多分辨率（5个常见宽高比：(1,1), (3,4), (4,3), (9,16), (16,9) ）及Min-Max token Strategy训练，而在训练视频时采用固定9：16的宽高比固定分辨率训练。

Skiparse Attention与Full Attention的区别在于前向过程中参与计算的token序列不同，所需要的权重变量则完全相同。因此，我们可以先用Full 3D Attention的Dense MMDiT做训练，并在训练充分后Fine-tune至Sparse MMDiT模式。

**Image-Stage-1:** 采用512张Ascend 910B进行训练。 我们采用随机初始化的Dense MMDiT在256^2px级别分辨率的图片上训练，开启多分辨率。学习率为1e-4，batch size为8096。在这个阶段我们总共训练了225k steps。

**Image-Stage-2:** 采用384张Ascend 910B进行训练。在384^px级别的图片上训练，开启多分辨率训练。学习率为1e-4，batch size为6144，共训练150k step。

**Image-Stage-3:** 采用256张Ascend 910B进行训练。固定288x512分辨率训练。学习率为1e-4，batch size为4096，共训练110k step。Dense MMDiT阶段训练完成。

**Image-Stage-4:** 采用256张Ascend 910B进行训练。采用Dense MMDiT的权重初始化SUV，其中skip connection采用零初始化，保证初始SUV权重能够推出非噪声图片。事实上，zero shot推理得到的图片具备一定的低频信息，我们验证了Dense DiT到SUV的finetune可以很快达成。该阶段固定分辨率为288x512，学习率为1e-4，batch size为4096，共训练约160k step。

#### Text-to-Video

在训练视频时，我们采用的宽高比固定为9：16，且并未采用视频图像联合训练，而是仅用视频数据做训练。以下训练均在512张Ascend 910B上完成。

**Video-Stage-1:** 继承Text-to-Image阶段得到的SUV权重，我们在57x288x512的视频上训练了大约40k step，学习率为6e-5，TP/SP并行度为2，学习率为6e-5，梯度累积次数为2， micro batch size为2，global batch size为1024。在这个阶段，我们采用的train fps为24，即大约57/24≈2.4s的视频内容。该阶段作为图片权重到视频权重迁移的第一个阶段，我们选择了较短的视频训练作为良好的初始化。

**Video-Stage-2: **我们同样在57x288x512的视频上训练45k step，学习率、TP/SP并行度和梯度累积设置保持不变，但是train fps更改为12，即对应的原视频长度为57/12≈4.8s的内容。该阶段旨在不增加序列长度的同时提高对时序的学习，为后续高帧数训练阶段做准备。

**Video-Stage-3:** 我们在121x288x512的视频上训练约25k step，学习率调整为4e-5、TP/SP并行度设置为4，梯度累积次数设置为2，micro batch size为4，global batch size为1024。在这个阶段我们重新采用train fps为24。

**Video-Stage-4:** 在121x576x1024的视频上共训练16k + 9k step，学习率分别为2e-5和1e-5，TP/SP并行度设置为4，梯度累积次数设置为4，micro batch size为1，global batch size为512。

**Video-Stage-5:** 我们选择数据中的高质量子集训练了5k step，学习率为1e-5，TP/SP并行度设置为4，梯度累积次数设置为4，micro batch size为1，global batch size为512。

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

Comming Soon...

### Future Work

目前，开源社区已经有与闭源商业版本相当性能的模型，如Wan2.1。鉴于算力和数据相比企业来说仍存在不足，后续Open-Sora Plan团队的改进方向为：

1、Latents Cache。

在Text2Video模型的训练过程中，训练数据需要经过变分自编码器（VAE）和文本编码器（Text Encoder）两个关键模块的处理，以实现对视频/图片和对应引导词的特征编码。这些编码后的特征数据作为模型训练的输入，参与后续训练流程。然而业界训练方案中，每个训练周期（Epoch）都需要对多模态训练数据集进行重复的特征编码计算，这不仅增加了额外的计算开销，还显著延长了整体训练时间。

具体而言，在传统的训练流程中，VAE和Text Encoder模型通常需要常驻于GPU显存中，以便在每个Epoch中实时执行特征编码任务。这种设计虽然确保了特征编码的实时性，但也导致了GPU显存占用率居高不下，成为制约训练效率的主要瓶颈之一。尤其是在处理大规模数据集或复杂模型时，显存资源的紧张会进一步加剧这一问题，限制了模型的参数量和训练速度。

为了解决上述问题，我们提出了一种特征值以查代算的优化方案。该方案的核心思想是将特征编码的计算过程与模型训练过程进行解耦。具体实现方式为：在训练前或首轮训练时计算耗时最高的引导词特征值，将其保存至外置高性能文件存储中。后续的训练过程中，模型可以直接从文件存储中读取这些预计算的特征数据，避免了重复的特征编码计算。这种设计不仅显著减少了计算资源的浪费，还大幅降低了GPU显存的占用率，使更多的显存资源可用于模型训练。

基于以下配置环境，统计使用特征数据存储前后的单个epoch及单个step的训练数据。实验表明，特征值存储方案**可缩短约30%多轮迭代训练时间，同时释放约20%显存资源。**

|  配置环境  |             详细信息              |
| :--------: | :-------------------------------: |
|    模型    | Open-Sora Plan v1.5.0 with 2B量级 |
|   数据集   |         100K图片及10K视频         |
| GPU服务器  |          8张Nvidia A800           |
| 特征值存储 |       华为OceanStor AI存储        |

测试数据：

| 训练阶段     | 测试类型         | Batch Size | 单Step耗时 | 单Epoch耗时 | 显存占用 |
| ------------ | ---------------- | ---------- | ---------- | ----------- | -------- |
| 低分辨率图片 | 通用方案         | 64         | 6.53s      | 21min12s    | 56GB     |
|              | 特征数据存储方案 | 64         | 4.10s      | 13min19s    | 40GB     |
|              | 通用方案         | 128        | 12.78s     | 20min39s    | 74GB     |
|              | 特征数据存储方案 | 128        | 7.81s      | 12min38s    | 50GB     |
| 低分辨率视频 | 通用方案         | 8          | 8.90s      | 26min23s    | 68GB     |
|              | 特征数据存储方案 | 8          | 7.78s      | 23min05s    | 51GB     |
| 高分辨率视频 | 通用方案         | 4          | 17s        | 101min      | 71GB     |
|              | 特征数据存储方案 | 4          | 16s        | 97min       | 57GB     |

2、更好的基于稀疏化attention or 线性attention预训练的DiT。在V1.3.0中，我们推出了社区中第一个基于稀疏attention预训练的DiT，并在V1.5.0版本中将其扩展为SUV架构，使稀疏DiT获得了与Dense DiT相当的模型性能。稀疏attention和线性attention在LLM领域已经获得了很大的成功，但在视频生成领域中的应用仍不够明显。在后续版本中，我们将进一步探索稀疏attention和线性attention在video generation领域的应用。

3、基于MoE的DiT。自Mixtral 8x7B发布以来，LLM领域通常会采用MoE的方式将模型scale至更大的参数量。目前开源视频模型的最大大小仅限于14B，相比于LLM领域上百B的参数量来说仍属于小模型。在DiT架构中引入MoE，以及MoE与稀疏attention和线性attention的结合，是Open-Sora Plan团队未来考虑的方向。

4、生成和理解统一的视频生成模型。3月份gpt-4o的更新让大家认识到了生成理解统一架构的生成模型能够获得与纯生成模型完全不同的能力。在视频领域，我们同样应该期待一个统一的生成模型能够为我们带来哪些惊喜。

5、更好的Image-to-Video模型。目前Image-to-Video领域仍基本遵循SVD范式和Open-Sora Plan v1.2.0起采用的Inpainting范式。这两种范式都需要在Text-to-Video模型权重的基础上进行长时间的finetune。从应用意义上看，Text-to-Video更接近于学术上的探索，而Image-to-Video则更贴近现实的生产环境。因此，Image-to-Video的更新范式也会是Open-Sora Plan团队未来的重点探索方向。
