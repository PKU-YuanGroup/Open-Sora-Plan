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

MindSpeed-MM是面向大规模分布式训练的昇腾多模态大模型套件，同时支持多模态生成及多模态理解，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、在线推理任务等特性。

---

## MindSpeed-MM大模型方案概览

当前MindSpeed-MM支撑大模型使用功能:

* [生成类多模态大模型](#jump1) 【昇腾】【NAIE】
* [理解类多模态大模型](#jump1) 【昇腾】【NAIE】【GTS】
* [预训练/全参微调/低参微调/在线推理](./examples/) 【昇腾】【NAIE】
* 数据工程： 多模数据预处理及加载/数据分桶策略 【昇腾】
* 分布式训练: [加速算法/融合算子/并行策略](#预训练加速算法与融合算子) 【昇腾】
* [昇腾工具链](#jump2): [Profiling采集](#jump2.1)【昇腾】

更多多模态模型持续研发中....

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

【现版本实测性能（硬件信息：Atlas 900 A2 PODc）】

下述列表中支持的模型，我们在各模型的`README`文件中提供了相应的使用说明，里面有详细的模型训练、推理、微调等流程

`模型`列中的超链接指向各模型的文件夹地址， `参数量`列中的超链接指向模型的社区资源地址

`认证`【Pass】表示已经过测试的模型，【Test】表示待测试模型

<table>
  <a id="jump1"></a>
  <caption>MindSpeed-MM模型列表</caption>
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
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.0">OpenSora 1.0</a></td>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.0.0">5.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 3.18 (Samples per Second)</td>
      <td> 2.04 (Samples per Second)</td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.2">OpenSora 1.2</a></td>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.2.0">5.2B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 18.90 (s/iter) </td>
      <td> 17.63 (s/iter) </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.2">OpenSoraPlan 1.2</a></td>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.2.0">10.8B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.42 (Samples per Second) </td>
      <td> 0.37 (Samples per Second) </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sdxl">SDXL</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 29.92  (FPS)</td>
      <td> 30.65 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 28.51 (FPS)</td>
      <td> 30.23 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 166.71 (FPS)</td>
      <td> 164.66 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sd3">SD3</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 17.08 (FPS)</td>
      <td> 17.51 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 16.57 (FPS)</td>
      <td> 16.36 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>Lora微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 122.47 (FPS)</td>
      <td> 120.32 (FPS)</td>
      <td> 【昇腾】【NAIE】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/kolors">Kolors</a></td>
      <td><a href="https://github.com/Kwai-Kolors/Kolors">2.6B</a></td>
      <td>推理</td>
      <td> 单机单卡</td>
      <td> FP16 </td>
      <td> / </td>
      <td> / </td>
      <td> 【NAIE】 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/llava1.5">LLaVA 1.5</a></td>
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
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl2.0">Intern-VL-2.0</a></td>
      <td><a href="https://github.com/OpenGVLab/InternVL2.0">8B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> 【昇腾】 </td>
      <td>【Test】</td>
    </tr>
      <td><a href="https://github.com/OpenGVLab/InternVL2.0">26B</a></td>
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
 <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/qwen2-vl">Qwen2-VL</a></td>
      <td><a href="https://qwen2.org/vl/">72B</a></td>
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
  <caption><a href="https://gitee.com/ascend/ModelZoo-PyTorch">ModelZoo仓内的多模态大模型</a>（Q4逐步迁移至MindSpeed-MM）</caption>
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
      <td>【Pass】</td>
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
      <td> 1046 (s)/50-200steps </td>
      <td> 847 (s)/50-200steps </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/OpenBMB/MiniCPM-V">8B</a></td>
      <td>Lora微调</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 603 (s)/50-200steps </td>
      <td> 490 (s)/50-200steps </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
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
      <td>【Pass】</td>
    </tr>
  </tbody>
</table>

---

## 预训练加速算法与融合算子

MindSpeed MM预训练支持多种分布式并行算法和融合算子，下表为一些各种加速特性对应的使能开关（不同模型请参考各自对应的使用手册：[examples](./examples/)）：

<table><thead>
  <tr>
    <th>使用场景</th>
    <th>特性名称</th>
    <th>具体参数</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="3">长序列并行</td>
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
    <td rowspan="4">显存优化</td>
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
    <td>融合算子</td>
    <td>Flash attention</td>
    <td>默认开启</td>
  </tr>
</tbody></table>

---

<a id="jump2"></a>

## MindSpeed-MM工具库

<a id="jump2.1"></a>

### 昇腾Profiling采集工具

MindSpeed-MM集成了昇腾profiling采集工具，以提供对模型运行情况的分析。该工具能够依照配置采集模型的算子、显存等关键信息，同时支持动静态两种采集方式，协助开发者分析模型瓶，并可根据实际场景需求选择使用。

  具体方法见 [README](./mindspeed_mm/tools/README.md) 的profiling章节

---

## 致谢

MindSpeed-MM 由华为公司的下列部门联合贡献 ：

* 昇腾计算产品部
* 公共开发部：NAIE
* 全球技术服务部：GTS

感谢来自社区的每一个PR，欢迎贡献 MindSpeed-MM

---

## 安全申明

[MindSpeed MM 安全申明](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/SECURITYNOTE.md)
