# InternVL2 使用指南

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

**注意事项:** 

  需要修改 mindspeed/core/transformer/dot_product_attention.py的65行，修改如下：
```python
def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params):
        # 注释下一行
        # attention_mask = get_attention_mask()
        if get_args().use_flash_attn:
            return dot_product_attention_forward(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)
        return fn(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)

    return wrapper
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B/tree/main)；
- [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)；


<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeeed-MM修改了部分原始网络的结构名称，使用`examples/internvl2/inernvl_convert_to_mm_ckpt.py`脚本对原始预训练权重进行转换。

以InternVL2-2b为例，修改`inernvl_convert_to_mm_ckpt.py`中的`load_dir`和`save_dir`如下

```python
  hg_ckpt_dir = 'OpenGVLab/InternVL2-2B' # huggingface权重目录
  mm_save_dir = 'ckpt/InternVL2-2B'  # 转换后保存目录
  pipeline_layer_index = None
  num_layers=24
```
启动脚本
```shell
  python examples/internvl2/inernvl_convert_to_mm_ckpt.py
```

同步修改`examples/internvl2/finetune_internvl2_2b.sh`中的`--load`参数.
```shell
  --load ckpt/InternVL2-2B
```

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压[InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data)数据集到`dataset/playground`目录下，以数据集ai2d为例，解压后的数据结构如下：

   ```
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
   ├── opensource
       ├── ai2d_train_12k.jsonl
   ```

---

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数
【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL2-2B为例，`data.json`进行以下修改
```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/ai2d_train_12k.jsonl",
          "data_folder": "dataset/playground/data/ai2d"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "OpenGVLab/InternVL2-2B",
          ...
      },
      ...
  },
  ...
}
```
【单机运行配置】

配置`examples/internvl2/finetune_internvl2.sh`参数如下

```shell
    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=29501
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以InternVL2-2B为例，启动微调。
```shell
    bash examples/internvl2/finetune_internvl2_2B.sh
```

<a id="jump5"></a>

## 推理
Coming Soon...
