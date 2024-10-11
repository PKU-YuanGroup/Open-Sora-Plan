# LLaVA1.5 使用指南

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
|  Python   |                3.10                 |
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
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp310-cp310m-linux_aarch64.whl

    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout 3da17d56
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

- [ViT-L-14-336px](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)：CLIPViT模型；

- [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/)： GPT模型；

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeeed-MM修改了部分原始网络的结构名称，因此需要使用如下脚本代码对下载的预训练权重进行转换。

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压image.zip得到[LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)数据集，获取数据结构如下：

   ```
   $LLaVA-Pretrain
   ├── blip_laion_cc_sub_558k.json
   ├── blip_laion_cc_sub_558k_meta.json
   ├── images
   ├── ├── 00000
   ├── ├── 00001
   ├── ├── 00002
   └── └── ...
   ```

---

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

需根据实际情况修改`model.json`和`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段。

【单机运行】

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=29501
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动预训练

```shell
    bash examples/llava1.5/pretrain_llava1_5.sh
```

<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作



<a id="jump5.2"></a>

#### 2. 配置参数

将准备好的权重传入到inference_llava.json中，更改其中的路径，包括from_pretrained，自定义的prompt可以传入到prompt字段中

<a id="jump5.3"></a>

#### 3. 启动推理

启动推理脚本

```shell
examples/llava1.5/inference_llava1_5.sh
```

---
