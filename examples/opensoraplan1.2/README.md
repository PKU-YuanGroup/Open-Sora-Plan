# OpenSoraPlan1.2 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

|    软件     | [版本](https://www.hiascend.com/zh/) |
|:---------:|:----------------------------------:|
|  Python   |                3.8                 |
|  Driver   |         RC3 商发版本          |
| Firmware  |         RC3 商发版本          |
|   CANN    |             RC3 商发版本             |
|   Torch   |            2.1.0            |
| Torch_npu |           2.1.0           |

<a id="jump1.1"></a>

#### 1. 仓库拉取

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

<a id="jump1.2"></a>

#### 2. 环境搭建

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

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [LanguageBind/Open-Sora-Plan-v1.2.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main)：CasualVAE模型和VideoDiT模型；

- [DeepFloyd/mt5-xxl](https://huggingface.co/google/mt5-xxl/)： MT5模型；

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeeed-MM修改了部分原始网络的结构名称，因此需要使用如下脚本代码对下载的预训练权重进行转换。

```python
import torch
from safetensors.torch import load_file

pretrained_checkpoint = torch.load("your pretrained path", map_location="cpu")
new_checkpoint = {}
for key in pretrained_checkpoint.keys():
    model_key = key.replace("transformer_blocks", "videodit_blocks")
    model_key = model_key.replace("attn1", "self_atten")
    model_key = model_key.replace("attn2", "cross_atten")
    model_key = model_key.replace("to_q", "proj_q")
    model_key = model_key.replace("to_k", "proj_k")
    model_key = model_key.replace("to_v", "proj_v")
    model_key = model_key.replace("to_out.0", "proj_out")
    model_key = model_key.replace("to_out.1", "dropout")
    new_checkpoint[model_key] = pretrained_checkpoint[key]

torch.save(new_checkpoint, "videodit.pth")
```

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压[pixabay_v2](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/pixabay_v2_tar)数据集，获取数据结构如下：

   ```
   $pixabay_v2
   ├── annotation.json
   ├── folder_01
   ├── ├── video0.mp4
   ├── ├── video1.mp4
   ├── ├── ...
   ├── ├── annotation.json
   ├── folder_02
   ├── folder_03
   └── ...
   ```

---

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

需根据实际情况修改`model_opensoraplan1_2.json`和`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段。

【单机运行】

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=29501
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

【多机运行】

```shell
    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8  #每个节点的卡数
    MASTER_ADDR="your master node IP"  #都需要修改为主节点的IP地址（不能为localhost）
    MASTER_PORT=29501
    NNODES=2  #集群里的节点数，以实际情况填写,
    NODE_RANK="current node id"  #当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动预训练

```shell
    bash examples/opensoraplan1.2/pretrain_opensoraplan1_2.sh
```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，需要在每个节点准备训练数据和模型权重

---

<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作

参考上述的权重下载及转换章节，需求的权重需要到huggingface中下载，以及参考上面的权重转换代码进行转换。
链接参考: [predict_model](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x480p) [VAE](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae) [tokenizer/text_encoder](https://huggingface.co/google/mt5-xxl/tree/main)

<a id="jump5.2"></a>

#### 2. 配置参数

将准备好的权重传入到inference_model_29x480x640.json中，更改其中的路径，包括from_pretrained，自定义的prompt可以传入到prompt字段中

<a id="jump5.3"></a>

#### 3. 启动推理

启动推理脚本

```shell
examples/opensoraplan1.2/inference_opensoraplan1_2.sh
```

---
