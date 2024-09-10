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

* [生成类多模态大模型](#) 【昇腾】
* [理解类多模态大模型](#) 【昇腾】
* [预训练](#j)/[全参微调](#)/[低参微调](#) 【昇腾】

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
  <thead>
    <tr>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>性能</th>
      <th>性能参考</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>生成类模型</th>
      <td colspan="9"></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">OpenSora 1.0</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sdxl">*B</a></td>
      <td>--</td>
      <th> 1x8 </th>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.0.0">*B</a></td>
      <td>--</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">OpenSora 1.2</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.2">*B</a></td>
      <td>--</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.2.0">*B</a></td>
      <td>--</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">OpenSoraPlan 1.2</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.2">*B</a></td>
      <td>--</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/hpcaitech/Open-Sora/tree/v1.2.0">*B</a></td>
      <td>--</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">CogvideoX</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">*B</a></td>
      <td>--</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideo">*B</a></td>
      <td>--</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sdxl">SDXL</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 24.7 </td>
      <td> 30.65 </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>预训练</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 23.24 </td>
      <td> 30.23 </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">3.5B</a></td>
      <td>SDXL 全参微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 164.66 </td>
      <td> 167.89 </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sd3">SD3</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 17.64 </td>
      <td> 17.51 </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>全参微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 15.63 </td>
      <td> 16.36 </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/eda36c4c286d281f216dfeb79e64adad3f85d37a">2B</a></td>
      <td>推理Lora微调</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 14.04 </td>
      <td> 14.82 </td>
      <td> *** </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <th>理解类模型</th>
      <td colspan="9"></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">Llava 1.5</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/llava">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/haotian-liu/LLaVA">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">InternVL 2.0</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/OpenGVLab/InternVL">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <th>原ModelZoo仓内的多模态大模型（后续将逐步整改至MindSpeed-MM）</th>
      <td colspan="9"></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">Cogvlm2</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvlm2">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/THUDM/CogVLM2">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">pllava</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/pllava">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/magic-research/PLLaVA">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">miniCPM 2.5</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/minicpm-v">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/OpenBMB/MiniCPM-V">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="">HunyuanDiT</a></td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/hunyuandit">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Tencent/HunyuanDiT">*B</a></td>
      <th>--</th>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> *** </td>
      <td> *** </td>
      <td> *** </td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>

---

具体的权重转换功能命令介绍见[examples/README.md](./examples/README.md)

---

## 安全申明

[MindSpeed MM 安全申明](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/SECURITYNOTE.md)
