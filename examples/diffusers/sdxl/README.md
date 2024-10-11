# Diffusers

<p align="left">
        <b>简体中文</b> |
</p>

- [SDXL](#jump1)
  - [模型介绍](#模型介绍)
  - [预训练](#预训练)
    - [环境搭建](#环境搭建)
    - [预训练](#jump2)
    - [性能](#性能)
  - [微调](#微调)
    - [环境搭建](#jump3)
    - [微调](#jump3.1)
    - [性能](#jump3.2)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
    - [性能](#jump4)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# Stable Diffusion XL

## 模型介绍

扩散模型（Diffusion Models）是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是 HuggingFace 发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频，甚至分子的3D结构。套件包含基于扩散模型的多种模型，提供了各种下游任务的训练与推理的实现。

- 参考实现：

  ```
  url=https://github.com/huggingface/diffusers
  commit_id=eda36c4c286d281f216dfeb79e64adad3f85d37a
  ```

## 预训练

### 环境搭建

【模型开发时推荐使用配套的环境版本】

|    软件     | [版本](https://www.hiascend.com/zh/) |
|:---------:|:----------------------------------:|
|  Python   |                3.8                 |
|  Driver   |         RC3 商发版本          |
| Firmware  |         RC3 商发版本          |
|   CANN    |             RC3 商发版本             |
|   Torch   |            2.1.0            |
| Torch_npu |           2.1.0           |

1. 软件与驱动安装

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 SDXL [GitHub参考实现](https://github.com/huggingface/diffusers)】或 [适配昇腾AI处理器的实现](https://gitee.com/ascend/ModelZoo-PyTorch.git) 或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    cd diffusers
    git reset --hard eda36c4c286d281f216dfeb79e64adad3f85d37a
    cp -r ../MindSpeed-MM/examples/diffusers/sdxl ./sdxl
    ```

    【主要代码路径】

    ```shell
    code_path=examples/text_to_image/
    ```

    3.2【安装 `{任务pretrain/train}_sdxl_deepspeed_{混精fp16/bf16}.sh` 需要[适配昇腾AI处理器的实现](https://gitee.com/ascend/ModelZoo-PyTorch.git)】

    转移 `collect_dataset.py` 与 `pretrain_model.py` 与 `train_text_to_image_sdxl_pretrain.py` 到 `examples/text_to_image/` 路径

    ```shell
    # Example, 需要修改.py名字进行三次任务
    cp ./sdxl/train_text_to_image_sdxl_pretrain.py ./examples/text_to_image/
    ```

    3.3【安装其余依赖库】

    ```shell
    pip install e .
    vim examples/text_to_image/requirements_sdxl.txt #修改torchvision版本：torchvision==0.16.0, torch==2.1.0
    pip install -r examples/text_to_image/requirements_sdxl.txt # 安装diffusers原仓对应依赖
    pip install -r sdxl/requirements_sdxl_extra.txt #安装sdxl对应依赖
    ```

<a id="jump2"></a>

### 预训练

1. 【准备预训练数据集】

    用户需自行获取并解压laion_sx数据集（目前数据集暂已下架，可选其他数据集）与[pokemon-blip-captions](https://gitee.com/hf-datasets/pokemon-blip-captions)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    修改`pretrain_sdxl_deepspeed_**16.sh`的dataset_name为`laion_sx`的绝对路径

    ```shell
    vim sdxl/pretrain_sdxl_deepspeed_**16.sh
    ```

    修改`train_sdxl_deepspeed_**16.sh`的dataset_name为`pokemon-blip-captions`的绝对路径

    ```shell
    vim sdxl/train_sdxl_deepspeed_**16.sh
    ```

    laion_sx数据集格式如下:

    ```shell
    laion_sx数据集格式如下
    ├── 000000000.jpg
    ├── 000000000.json
    ├── 000000000.txt
    ```

    pokemon-blip-captions数据集格式如下:

    ```shell
    pokemon-blip-captions
    ├── dataset_infos.json
    ├── README.MD
    └── data
          └── train-001.parquet
    ```

    > **说明：**
    >该数据集的训练过程脚本只作为一种参考示例。
    >
  
2. 【配置 SDXL 预训练脚本与预训练模型】

    联网情况下，预训练模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[sdxl-base模型](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) `model_name`模型与[sdxl-vae模型](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) `vae_name`

    ```bash
    export model_name="stabilityai/stable-diffusion-xl-base-1.0" # 预训练模型路径
    export vae_name="madebyollin/sdxl-vae-fp16-fix" # vae模型路径
    ```

    获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径

    ```bash
    scripts_path="./sdxl" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-xl-base-1.0" # 预训练模型路径
    vae_name="madebyollin/sdxl-vae-fp16-fix" # vae模型路径
    dataset_name="laion_sx" # 数据集路径
    batch_size=4
    max_train_steps=2000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="${scripts_path}/pretrain_${mixed_precision}_accelerate_config.yaml"
    ```

    修改bash文件中`accelerate`配置下`train_text_to_image_sdxl_pretrain.py`的路径（默认路径在diffusers/sdxl/）

    ```bash
    accelerate launch --config_file ${config_file} \
      ${scripts_path}/train_text_to_image_sdxl_pretrain.py \  #如模型根目录为sdxl则无需修改
    ```

    修改`pretrain_fp16_accelerate_config.yaml`的`deepspeed_config_file`的路径:

    ```bash
    deepspeed_config_file: ./sdxl/deepspeed_fp16.json # deepspeed JSON文件路径
    ```

3. 【启动 SDXL 预训练脚本】

    本任务主要提供**混精fp16**和**混精bf16**两种**8卡**训练脚本，默认使用**deepspeed**分布式训练。

    **pretrain**模型主要来承担第二阶段的文生图的训练
    **train**模型主要来承担第一阶段的文生图的训练功能

    ```shell
    bash sdxl/pretrain_sdxl_deepspeed_**16.sh
    bash sdxl/train_sdxl_deepspeed_**16.sh
    ```

### 性能

#### 吞吐

SDXL 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| 竞品A | 8p | SDXL_train_bf16  |  30.65 |     4      | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_train_bf16  | 29.92 |     4      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | SDXL_train_fp16 |  30.23 |     4      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_train_fp16 | 28.51 |     4      | fp16 | 2.1 | ✔ |
| 竞品A | 8p | SDXL_pretrain_bf16  |  21.14 |     4      | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_pretrain_bf16  | 19.79 |     4      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | SDXL_pretrain_fp16 |  20.77 |     4      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_pretrain_fp16 | 19.67 |     4      | fp16 | 2.1 | ✔ |

## 微调

<a id="jump3"></a>

### 环境搭建

#### LORA微调

   > **说明：**
   > 数据集同预训练的`pokemon-blip-captions`，请参考预训练章节。
   >

  ```shell
  sdxl/finetune_sdxl_lora_deepspeed_fp16.sh
  ```

#### Controlnet微调

   1. 联网情况下，数据集会自动下载。
   2. 无网络情况下，用户需自行获取fill50k数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径，以及需要修改里面fill50k.py文件。

   ```shell
   sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh
   ```

   3. 参考如下修改controlnet/train_controlnet_sdxl.py, 追加trust_remote_code=True

   ```shell
   dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True
          )
   ```

   > **注意：**
   >需要修改数据集下面的fill50k.py文件中的57到59行，修改示例如下:
>
   > ```python
   > metadata_path = "数据集路径/fill50k/train.jsonl"
   > images_dir = "数据集路径/fill50k"
   > conditioning_images_dir = "数据集路径/fill50k"
   >```
>
   fill50k数据集格式如下:

   ```
   fill50k
   ├── images
   ├── conditioning_images
   ├── train.jsonl
   └── fill50k.py
   ```

   > **说明：**
   >该数据集的训练过程脚本只作为一种参考示例。

#### 全参微调

   > **说明：**
   > 数据集同Lora微调，请参考Lora章节。
   >
  【获取预训练模型】

   获取[sdxl-base模型](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) `model_name`模型与[sdxl-vae模型](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) `vae_name`。
  
   获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径。

   ```shell
   sdxl/finetune_sdxl_deepspeed_fp16.sh
   ```

   > **说明：**
   > 预训练模型同预训练，请参考预训练章节。
   >

<a id="jump3.1"></a>

### 微调

   【运行微调的脚本】

    ```shell
    # 单机八卡微调
    bash sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh      #8卡deepspeed训练 sdxl_controlnet fp16
    bash sdxl/finetune_sdxl_lora_deepspeed_fp16.sh            #8卡deepspeed训练 sdxl_lora fp16
    bash sdxl/finetune_sdxl_deepspeed_fp16.sh        #8卡deepspeed训练 sdxl_finetune fp16
    ```

<a id="jump3.2"></a>

### 性能

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| 竞品A | 8p |    LoRA    | 31.74 |     7      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |    LoRA    | 26.40 |     7      | fp16 | 2.1 | ✔ |
| 竞品A | 8p | Controlnet | 32.44  |     5      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | Controlnet | 29.98 |     5      | fp16 | 2.1 | ✔ |
| 竞品A | 8p |  Finetune  | 164.66 |     24     | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |  Finetune  | 166.71 |     24     | fp16 | 2.1 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

 【运行推理的脚本】

- 单机单卡推理
- 调用推理脚本

  ```shell
  python sdxl/sdxl_text2img_lora_infer.py        # 混精fp16 文生图lora微调任务推理
  python sdxl/sdxl_text2img_controlnet_infer.py  # 混精fp16 文生图controlnet微调任务推理
  python sdxl/sdxl_text2img_infer.py             # 混精fp16 文生图全参微调任务推理
  python sdxl/sdxl_img2img_infer.py              # 混精fp16 图生图微调任务推理
  ```

<a id="jump4"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:---:|:---:|:---:|
| 竞品A | 8p |    文生图lora    | 1.45 |  fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |    文生图lora    | 2.61 |  fp16 | 2.1 | ✔ |
| 竞品A | 8p | 文生图controlnet | 1.41  |  fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | 文生图controlnet | 2.97 |  fp16 | 2.1 | ✔ |
| 竞品A | 8p |  文生图全参  | 1.55 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |  文生图全参  | 3.02 | fp16 | 2.1 | ✔ |
| 竞品A | 8p |  图生图  | 3.56 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |  图生图  | 3.94 | fp16 | 2.1 | ✔ |

## 引用

### 公网地址说明

[代码涉及公网地址](/MindSpeed-MM/docs/public_address_statement.md)参考 docs/public_address_statement.md
