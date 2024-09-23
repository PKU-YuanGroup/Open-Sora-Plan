  <p align="center"> <img src="sources/images/logo.png" height="103px" width="700px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

MindSpeed-MM旨在为华为 [昇腾芯片](https://www.hiascend.com/) 上提供端到端的多模态大模型训练解决方案, 包含模型，加速，以及下游任务。

---

## MindSpeed-MM大模型方案概览

当前MindSpeed-MM支撑大模型使用功能:

* [生成类多模态大模型](#jump1) 【昇腾】【NAIE】
* [理解类多模态大模型](#jump1) 【昇腾】【NAIE】【GTS】
* [预训练/全参微调/低参微调/在线推理](./examples/) 【昇腾】【NAIE】
* 数据工程： 多模数据预处理及加载/动态分辨率数据分桶策略 【昇腾】
* 分布式训练: [加速算法/融合算子/并行策略](#jump2) 【昇腾】
* 昇腾工具链: [Profiling采集](#jump3)【昇腾】

各关键多模态模型等持续研发中....

---

## 配套版本与支持模型

【版本配套环境】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          Driver           |         RC3 商发版本          |
|         Firmware          |         RC3 商发版本          |
|           CANN            |             RC3 商发版本             |
|           Torch           |            2.1.0            |
|         Torch_npu         |           2.1.0           |

【多模态大模型介绍】

MindSpeed-MM 通过模型并行与数据并行来训练各类多模态大模型，通过端到端的训练来进行文生图、图生图、文生视频、图生视频等。训练与测量所涵盖的操作包括数据加载、优化器步骤、通信、与日志记录。

【现版本实测性能（硬件信息：Atlas 900 A2 PODc）】

下述列表中支持的模型，我们在[examples/README.md](./examples/README.md)中提供了相应的使用说明，里面有详细的模型训练、推理、微调、评估流程

`模型`列中的超链接指向各模型的文件夹地址， `参数量`列中的超链接指向模型的社区资源地址

`认证`【Pass】表示经过昇腾官方版本测试的模型，【Test】表示待测试模型

<table>
  <caption>MindSpeed-MM模型列表</caption>
  <thead>
    <tr>
      <span id="jump1"><th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.0">OpenSora 1.0</a></td>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.0.0">5.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 3.18 (FPS) </td>
      <td> 2.04 (FPS) </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.2">OpenSora 1.2</a></td>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.2.0">5.2B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.2">OpenSoraPlan 1.2</a></td>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.2.0">10.8B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 29 (FPS) </td>
      <td> 33 (FPS) </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sdxl">SDXL</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 24.7 (FPS)</td>
      <td> 30.65 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 23.24 (FPS)</td>
      <td> 30.23 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>SDXL 全参微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 164.66 (FPS)</td>
      <td> 167.89 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sd3">SD3</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 17.64 (FPS)</td>
      <td> 17.51 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 15.63 (FPS)</td>
      <td> 16.36 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>推理Lora微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 14.04 (FPS)</td>
      <td> 14.82 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/llava">LLaVA 1.5</a></td>
      <td><a href="https://github.com/haotian-liu/LLaVA">7B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl1.5">Intern-VL-1.5</a></td>
      <td><a href="https://github.com/OpenGVLab/InternVL">26B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl2.0">Intern-VL-2.0</a></td>
      <td><a href="https://github.com/OpenGVLab/InternVL">26B</a></td>
      <td>/</td>
      <td> /</td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【Coming Soon】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideo">5B</a></td>
      <td>/</td>
      <td> /</td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【Coming Soon】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/qwen2-vl">Qwen2-VL</a></td>
      <td><a href="https://qwen2.org/vl/">7B</a></td>
      <td>/</td>
      <td> /</td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【Coming Soon】</td>
    </tr>
    </tbody>
</table>

<table>
  <caption><a href="https://gitee.com/ascend/ModelZoo-PyTorch">原ModelZoo仓内的多模态大模型</a>（后续将逐步迁移至MindSpeed-MM）</caption>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/CogVLM2">CogVLM-2</a></td>
      <td><a href="https://github.com/THUDM/CogVLM2">8B</a></td>
      <td>微调</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 3.9 (s/it) </td>
      <td> 3.3 (s/it) </td>
      <td> 【GTS】 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/PLLaVA">PLLaVA</a></td>
      <td><a href="https://github.com/magic-research/PLLaVA">7B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.841 (s/step) </td>
      <td> 0.935 (s/step) </td>
      <td> 【NAIE】 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/magic-research/PLLaVA">7B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> FP32 </td>
      <td> 0.935 (s/step) </td>
      <td> 1.08 (s/step) </td>
      <td>【NAIE】 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/MiniCPM-V">miniCPM 2.5</a></td>
      <td><a href="https://github.com/OpenBMB/MiniCPM-V">8B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1046 (s) </td>
      <td> 847 (s) </td>
      <td> 【昇腾】 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/OpenBMB/MiniCPM-V">8B</a></td>
      <td>Lora微调</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 603 (s) </td>
      <td> 490 (s) </td>
      <td> 【昇腾】 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/HunyuanDiT">HunYuanDiT</a></td>
      <td><a href="https://github.com/Tencent/HunyuanDiT">1.5B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1099.5 (ms/step) </td>
      <td> 1059.3 (ms/step) </td>
      <td> 【昇腾】 </td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>

---

## <span id="jump2"> 预训练加速算法与融合算子

MindSpeed MM预训练支持张量并行、流水线并行等多种加速算法和融合算子，下表为各种加速特性对应的使能开关：

<table><thead>
  <tr>
    <th>使用场景</th>
    <th>特性名称</th>
    <th>具体参数</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="4">长序列并行</td>
    <td>序列并行</td>
    <td>--sequence-parallel</td>
  </tr>
  <tr>
    <td>分布式优化器</td>
    <td>--use-distributed-optimizer</td>
  </tr>
  <tr>
    <td>长序列并行</td>
    <td>--context-parallel-size</td>
  </tr>
  <tr>
    <td>多并行方案</td>
    <td>--context-parallel-algo</td>
  </tr>
  <tr>
    <td rowspan="7">显存优化</td>
    <td>BF16 参数副本复用</td>
    <td>--reuse-fp32-param</td>
  </tr>
  <tr>
    <td>激活函数重计算</td>
    <td>--recompute-activation-function</td>
  </tr>
 <tr>
    <td>Swap Attention</td>
    <td>--swap-attention</td>
  </tr>
  <tr>
    <td>重计算程度</td>
    <td>--recompute-granularity</td>
  </tr>
  <tr>
    <td>重计算层数</td>
    <td>--recompute-num-layers</td>
  </tr>
  <tr>
    <td>重计算方法</td>
    <td>--recompute-method</td>
  </tr>
  <tr>
    <td>PP-Stage重计算</td>
    <td>--enable-recompute-layers-per-pp-rank</td>
  </tr>
  <tr>
    <td rowspan="5">融合算子</td>
    <td>Flash attention</td>
    <td>--use-flash-attn</td>
  </tr>
  <tr>
    <td>Fused rms_norm</td>
    <td>--use-fused-rmsnorm</td>
  </tr>
  <tr>
    <td>Fused swiglu</td>
    <td>--use-fused-swiglu</td>
  </tr>
  <tr>
    <td>Fused rotary position embedding</td>
    <td>--use-fused-rotary-pos-emb</td>
  </tr>
  <tr>
    <td>Sliding window attention</td>
    <td>--sliding-window</td>
  </tr>
  <tr>
    <td rowspan="5">通信</td>
    <td>梯度reduce通算掩盖</td>
    <td>--overlap-grad-reduce</td>
  </tr>
  <tr>
    <td>权重all-gather通算掩盖</td>
    <td>--overlap-param-gather</td>
  </tr>
  <tr>
    <td>MC2</td>
    <td>--use-mc2</td>
  </tr>
  <tr>
    <td>MLP通信隐藏</td>
    <td>--use-pipe-experts</td>
  </tr>
  <tr>
    <td>计算通信并行 CoC</td>
    <td>--use-ascend-coc</td>
  </tr>
</tbody></table>

**注意事项**

具体的预训练方法见[examples](./examples/)

---

## <span id="jump3"> 基于昇腾芯片采集Profiling数据

MindSpeed-MM支持基于昇腾芯片采集profiling数据，以提供对模型运行情况的分析，主要API如下：

```bash
--profile                        # 打开profiling采集数据开关
--profile-step-start  10          # 指定开启采集数据的步骤，未配置时默认为10
--profile-step-end 12             # 指定结束采集数据的步骤，未配置时默认为12，实际采集步数为 end-start，不包含end
--profile-ranks 0 1 2 3 4        # 指定采集数据的卡号，默认为-1，表示采集所有rank的profiling数据，可以设置为 0 1 2 3 4 5 6 7 8 9 列表指定每个rank在单机/集群中的全局值
--profile-level level1           # 数据采集水平，level0, 1, 2, 级别越高采集信息越多，默认为level0
--profile-with-cpu               # 是否采集CPU数据，加入参数采集
--profile-with-stack             # 采集指令运行堆栈，加入参数采集
--profile-with-memory            # 是否采集内存，加入参数采集，配置开关时需打开--profile-with-cpu
--profile-record-shapes          # 是否采集计算shape，加入参数采集
--profile-save-path ./profile_dir    # profiling数据采集保存路径
```

---

## 致谢

MindSpeed MM 由华为公司的下列部门联合贡献 ：

* 昇腾计算产品部
* 公共开发部：NAIE
* 全球技术服务部：GTS

感谢来自社区的每一个PR，欢迎贡献 MindSpeed MM

---

## 安全申明

[MindSpeed MM 安全申明](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/SECURITYNOTE.md)
