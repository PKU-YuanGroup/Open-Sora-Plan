# Diffusers

<p align="left">
        <b>简体中文</b> |
</p>

- [SD3](#StableDiffusionXL)
  - [模型介绍](#模型介绍)
  - [微调](#微调)
    - [环境搭建](#环境搭建)
    - [微调](#微调)
    - [性能](#性能)
  - [推理](#推理)
    - [环境搭建](#环境搭建)
    - [推理](#推理)
    - [性能](#性能)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

# Stable Diffusion 3

## 模型介绍

扩散模型（Diffusion Models）是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是 HuggingFace 发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频，甚至分子的3D结构。套件包含基于扩散模型的多种模型，提供了各种下游任务的训练与推理的实现。

- 参考实现：

  ```
  url=https://github.com/huggingface/diffusers
  commit_id=eda36c4c286d281f216dfeb79e64adad3f85d37a
  ```

## 微调

### 环境搭建

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
    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout core_r0.6.0
    pip install -r requirements.txt
    pip3 install -e .
    cd ..

    # 克隆仓库
    git clone https://gitee.com/ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 SD3 [GitHub参考实现](https://github.com/huggingface/diffusers) 或 [适配昇腾AI处理器的实现](https://gitee.com/ascend/ModelZoo-PyTorch.git) 或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    git reset --hard eda36c4c286d281f216dfeb79e64adad3f85d37a
    cp -r MindSpeed-MM/examples/diffusers/sd3 diffusers/sd3
    ```

    【主要代码路径】

    ```
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    cd diffusers
    pip install e .
    vim examples/text_to_image/requirements_sdxl.txt #修改torchvision版本：torchvision==0.16.0, torch==2.1.0
    pip install -r examples/dreambooth/requirements_sd3.txt # 安装对应依赖
    ```

## 微调

1. 【准备微调数据集】

    用户需自行获取并解压[pokemon-blip-captions](https://gitee.com/hf-datasets/pokemon-blip-captions)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    ```shell
    vim sd3/finetune_sd3_dreambooth_deepspeed_**16.sh
    vim sd3/finetune_sd3_dreambooth_fp16.sh
    ```

    ```shell
    dataset_name="pokemon-blip-captions" # 数据集 路径
    ```

   - pokemon-blip-captions数据集格式如下:

    ```
    pokemon-blip-captions
    ├── dataset_infos.json
    ├── README.MD
    └── data
          ├── dataset_infos.json
          └── train-001.parquet
    ```

    - 只包含图片的训练数据集，如非deepspeed脚本使用训练数据集dog:[下载地址](https://huggingface.co/datasets/diffusers/dog-example)，在shell启动脚本中将`input_dir`参数设置为本地数据集绝对路径>

    ```shell
    input_dir="dog" # 数据集路径
    ```

    ```
    dog
    ├── alvan-nee-*****.jpeg
    ├── alvan-nee-*****.jpeg
    ```

    > **说明：**
    >该数据集的训练过程脚本只作为一种参考示例。
    >

2. 【配置 SD3 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[sdxl-base模型](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) `model_name`模型

    ```bash
    export model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    scripts_path="./sd3" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型路径
    dataset_name="pokemon-blip-captions" 
    batch_size=4
    max_train_steps=2000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="${scripts_path}/${mixed_precision}_accelerate_config.yaml"
    ```

    数据集选择：如果选择默认[原仓数据集](https://huggingface.co/datasets/diffusers/dog-example),需修改两处`dataset_name`为`input_dir`：

    ```shell
    input_dir="dog"

    # accelerator 修改 --dataset_name=#dataset_name
    --instance_data_dir=$input_dir
    ```

    修改`fp16_accelerate_config.yaml`的`deepspeed_config_file`的路径:

    ```shell
    vim sd3/fp16_accelerate_config.yaml
    # 修改：
    deepspeed_config_file: ./sd3/deepspeed_fp16.json # deepspeed JSON文件路径
    ```

3. 【启动 SD3 微调脚本】

    本任务主要提供**混精fp16**和**混精bf16**两种**8卡**训练脚本，使用与不使用**deepspeed**分布式训练。

    ```shell
    sd3/finetuning_sd3_dreambooth_deepspeed_**16.sh #使用deepspeed
    sd3/finetune_sd3_dreambooth_fp16.sh #无使用deepspeed
    ```

### 性能

#### 吞吐

SD3 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Dreambooth-全参微调  |   17.64 |     4      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | Dreambooth-全参微调  |  17.51 |     4      | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | Dreambooth-全参微调 |  15.63 |     4      | fp16 | 2.1 | ✔ |
| 竞品A | 8p | Dreambooth-全参微调 |   16.36 |     4      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | Dreambooth-全参微调 | 1.34  | 1 | fp16 | 2.1 | ✘ |
| 竞品A | 8p | Dreambooth-全参微调 | 1.51 | 1 | fp16 | 2.1 | ✘ |

## 推理

### 环境搭建

1. 【安装与获取模型与数据集】

    获取[sdxl-base模型](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) 后，获取模型后在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径。

    ```shell
    sd3/finetune_sd3_dreambooth_lora_fp16.sh
    ```

    > **说明：**
    > 预训练模型同微调，请参考微调章节。
    > 安装与步骤同微调，请参考微调章节。
    > 数据集同微调，请参考微调章节。

2. 【启动 SD3 推理脚本】

    2.1 进入推理脚本目录

    ```shell
    cd sd3
    ```

    2.2 运行推理的脚本

    调用推理脚本，图生图推理脚本需先准备图片到当前路径下：[下载地址](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png)

    ```shell
    python infer_sd3_img2img_fp16.py   # 单卡推理，文生图
    python infer_sd3_text2img_fp16.py  # 单卡推理，图生图
    ```

### 推理

1. 【启动 SD3 推理脚本】

    ```shell
    sd3/finetune_sd3_dreambooth_lora_fp16.sh
    ```

### 性能

#### 吞吐

SD3 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 | 任务 | FPS | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc |8p | DreamBooth-LoRA | 14.04 | 8 | fp16 | 2.1 | ✘ |
| 竞品A | 8p | DreamBooth-LoRA | 15.04 | 8 | fp16 | 2.1 | ✘ |

## 使用基线数据集进行评估

## 引用

### 公网地址说明

代码涉及公网地址参考 public_address_statement.md
