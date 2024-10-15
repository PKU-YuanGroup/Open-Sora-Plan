# CogVideoX 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#环境安装)
  - [仓库拉取](#仓库拉取)
  - [环境搭建](#环境搭建)
- [权重下载及转换](#权重下载及转换)
  - [权重下载](#权重下载)
- [数据集准备及处理](#数据集准备及处理)
- [预训练](#预训练)
- [推理](#推理)
  - [准备工作](#准备工作)
  - [配置参数](#配置参数)
  - [启动推理](#启动推理)

---

## <span id="jump1"> 环境安装

【模型开发时推荐使用配套的环境版本】

|    软件     | [版本](https://www.hiascend.com/zh/) |
|:---------:|:----------------------------------:|
|  Python   |                3.8                 |
|  Driver   |                RC3 商发版本                |
| Firmware  |                RC3 商发版本                |
|   CANN    |                RC3 商发版本                |
|   Torch   |               2.1.0                |
| Torch_npu |               2.1.0                |

#### <span id="jump1.1"> 仓库拉取

```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
    mkdir logs
    mkdir dataset
    mkdir ckpt
```

#### <span id="jump1.2"> 环境搭建

```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 5dc1e83b
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -e .
```

---

## <span id="jump2"> 权重下载及转换

#### <span id="jump2.1"> 权重下载

从Huggingface下载开源模型权重

若使用5B模型：

模型地址为：https://huggingface.co/THUDM/CogVideoX-5b/tree/main

若使用2B模型：

模型地址为：https://huggingface.co/THUDM/CogVideoX-2b/tree/main


---

## 数据集准备及处理

#### 数据集下载

Coming Soon

---

## 预训练

Coming Soon

---

## <span id="jump5">推理

#### <span id="jump5.1"> 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

#### <span id="jump5.2"> 配置参数

检查如下配置是否完成

| 配置文件 |               修改字段               |                修改说明                 |
|------|:--------------------------------:|:-----------------------------------:|
|  examples/cogvideox/inference_model.json    |         from_pretrained          |            修改为下载的权重所对应路径            |
|  examples/cogvideox/inference_model.json    |          model_version           | 5B：cogvideox_5b<br/>2B：cogvideox_2b |
|  examples/cogvideox/inference_model.json    |         use_dynamic_cfg          |        5B：true<br/>2B：false         |
|   examples/cogvideox/samples_prompts.txt   |               文件内容               |      可自定义自己的prompt，一行为一个prompt      |

#### <span id="jump5.3"> 启动推理

```bash
bash examples/cogvideox/inference_cogvideox.sh
```

---
