# Whisper 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [数据集准备及处理](#jump2)
  - [数据集下载](#jump2.1)
- [预训练](#jump3)
  - [准备工作](#jump3.1)
  - [配置参数](#jump3.2)
  - [启动预训练](#jump3.3)

---

<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

|    软件     | [版本](https://www.hiascend.com/zh/) |
|:---------:|:----------------------------------:|
|  Python   |                3.10                |
|  Driver   |                RC3 商发版本                |
| Firmware  |                RC3 商发版本                |
|   CANN    |                RC3 商发版本                |
|   Torch   |               2.1.0                |
| Torch_npu |                2.1.0                |

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
    pip install torch-2.1.0-cp310-cp310-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp310-cp310-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp310-cp310-linux_aarch64.whl

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout 5dc1e83b
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install librosa
    conda install -c conda-forge libsndfile
    pip install -e .
```


<a id="jump2"></a>

## 数据集准备及处理

<a id="jump2.1"></a>

#### 1. 数据集与权重下载

用户需自行获取mozilla-foundation/common_voice_11_0数据集与openai/whisper-large-v3权重，
获取数据结构如下：

   ```
   $common_voice_11_0
   ├── audio
   ├── ├── hi
   ├── ├── ├── train
   ├── ├── ├── ├── hi_train_0.tar
   ├── ├── ├── test
   ├── ├── ├── ...
   ├── ├── en
   ├── ├── ...
   ├── transcript
   ├── ├── hi
   ├── ├── ├── train.tsv
   ├── ├── ├── test.tsv
   ├── ├── ├── ...
   ├── ├── en
   ├── ├── ...
   ├── common_voice_11_0.py
   ├── count_n_shard.py
   └── ...
   ```

获取权重结构如下：

   ```
   $whisper-large-v3
   ├── config.json
   ├── pytorch_model.bin
   ├── tokenizer.json
   └── ...
   ```

---

<a id="jump3"></a>

## 预训练

<a id="jump3.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**，详情可查看对应章节

 <a id="jump3.2"></a>

#### 2. 配置参数

需根据实际情况修改`model.json`和`data.json`中的权重和数据集路径，包括`dataset_name_or_path`、`processor_name_or_path`等字段

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

<a id="jump3.3"></a>

#### 3. 启动预训练

```shell
    bash examples/whisper/pretrain_whisper.sh
```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，需要在每个节点准备训练数据
