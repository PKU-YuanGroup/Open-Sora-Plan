# Diffusers

<p align="left">
        <b>简体中文</b> |
</p>

- [SDXL](#StableDiffusionXL)
  - [模型介绍](#模型介绍)
  - [预训练](#预训练) 
    - [环境搭建](#环境搭建)
    - [性能](#性能)
      - [吞吐](#吞吐)
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


# Diffusers 0.30.0 for Pytorch

## 模型介绍

扩散模型（Diffusion Models）是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是 HuggingFace 发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频，甚至分子的3D结构。套件包含基于扩散模型的多种模型，提供了各种下游任务的训练与推理的实现。

- 参考实现：

  ```
  url=https://github.com/huggingface/diffusers
  commit_id=eda36c4c286d281f216dfeb79e64adad3f85d37a
  ```

## 环境搭建

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
    cd MindSpeed-MM
    mkdir logs
    mkdir dataset
    mkdir ckpt

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout core_r0.6.0
    pip install -r requirements.txt
    pip3 install -e .
    cd ..
    ```

3. 模型搭建

    3.1 【下载 SDXL [GitHub参考实现](https://github.com/huggingface/diffusers) 或 [适配昇腾AI处理器的实现](https://gitee.com/ascend/ModelZoo-PyTorch.git) 或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    code_path=examples/text_to_image/
    ```

    or 

    ```shell
    git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/diffusion/
    ```

    or 

    ```python
    # 安装本地diffusers代码仓
    pip install diffusers==0.30.0                                               
    ```

    3.2【安装 `{任务pretrain/train}_sdxl0.30.0_deepspeed_{混精fp16/bf16}.sh` 适配文件： 只有[GitHub参考实现](https://github.com/huggingface/diffusers)需要】

    下载并安装 [collect_dataset.py](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/diffusion/diffusers0.25.0/examples/text_to_image/collect_dataset.py) 与 [pretrain_model.py](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/diffusion/diffusers0.25.0/examples/text_to_image/pretrain_model.py) 到 `examples/text_to_image/` 路径
    
    3.3【安装其余依赖库】

    ```shell
    pip install -r requirements.txt
    pip install -r examples/text_to_image/requirements_sdxl.txt # 安装对应依赖
    pip install -r requirements_sdxl_extra.txt #安装对应依赖
    ```

4. 训练与预训练

    4.1 【准备预训练数据集】

    用户需自行获取并解压LAION_5B数据集（目前数据集暂已下架，可选其他数据集），并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径
    
    ```shell
    dataset_name="laion5b" # 数据集 路径
    ```

    4.2 【配置 SDXL 预训练脚本】

    联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载

    获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径

    ```shell
    scripts_path="./sdxl_pretrain" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-xl-base-1.0" # 预训练模型
    vae_name="madebyollin/sdxl-vae-fp16-fix" #vae模型
    dataset_name="laion5b" 
    batch_size=4
    max_train_steps=2000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="${scripts_path}/pretrain_${mixed_precision}_accelerate_config.yaml"
    ```

    4.3 【source CANN环境】

    ```shell
    CANN_INSTALL_PATH_CONF='etc/Ascend/ascend_cann_install.info'

    if [ -f $CANN_INSTALL_PATH_CONF ]; then 
        CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cat -d "=" -f 2)
    else
      CANN_INSTALL_PATH="/usr/local/Ascend"
    fi

    if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ]; 
    then 
    source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
    else 
    source ${CANN_INSTALL_PATH}/nnae/set_env.sh
    fi
    ```

    4.4 【启动 SDXL 预训练脚本】

    本任务主要提供**混精fp16**和**混精bf16**两种**8卡**训练脚本，默认使用**deepspeed**分布式训练。

    **pretrain**模型主要来承担第二阶段的文生图的训练
    **train**模型主要来承担第一阶段的文生图的训练功能

    ```shell
    pretrain_sdxl_deepspeed_**16.sh
    train_sdxl_deepspeed_**16.sh
    ```
   
## 性能

### 吞吐

SDXL 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备   | 模型        | 迭代数  | 样本吞吐 (f/p/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 浮点计算数 (TFLOPs/s) |
|------|-----------|------|--------------------|----------------------|-----------------|------------------|
| NPUs | SDXL_pretrain_bf16  | * | .               | .                 | .            | .            |
| 参考 | SDXL_pretrain_bf16  | * | .               | .                 | .            | .            |
| NPUs | SDXL_pretrain_fp16 | * | .               | .                 | .            | .           |
| 参考 | SDXL_pretrain_fp16 | * | .               | .                 | .           | .           |
| NPUs | SDXL_train_bf16  | * | .               | .                 | .            | .            |
| 参考 | SDXL_train_bf16  | * | .               | .                 | .            | .            |
| NPUs | SDXL_train_fp16 | * | .               | .                 | .            | .           |
| 参考 | SDXL_train_fp16 | * | .               | .                 | .           | .           |


## 微调
### 环境搭建
1、软件与驱动安装
```
    # python3.10
    conda create -n SDXL python=3.10
    conda activate SDXL

    # 安装 torch 和 torch_npu
    pip install torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0.post6.dev20240716-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl


    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2、克隆仓库到本地服务器
```
    git clone https://gitee.com/ascend/MindSpeed-MM.git
    git clone https://github.com/huggingface/diffusers.git
    cp -r MindSpeed-MM/examples/diffusers/SDXL diffusers/examples
```

3、安装加速库和torch依赖包
```
    # 安装加速库
    cd diffusers
    pip install -e .
    vim examples/text_to_imge/requirements_sdxl.txt #修改torchvision版本：torchvision==0.16.0
    pip install -r examples/text_to_image/requirements_sdxl.txt
    
    # 安装torch依赖包
    pip install deepspeed
    pip install decorator==4.4.2
    pip install scipy=1.10.1
```

### 微调
#### 准备数据集
##### LORA微调
   1. 联网情况下，数据集会自动下载。
   2. 无网络情况下，用户需自行获取pokemon-blip-captions数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径。

   ```shell
   examples/SDXL/sdxl_text2img_lora_deepspeed.sh
   ```

   pokemon-blip-captions数据集格式如下:
   ```
   pokemon-blip-captions
   ├── dataset_infos.json
   ├── README.MD
   └── data
        ├── dataset_infos.json
        └── train-001.parquet
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
##### Controlnet微调

   1. 联网情况下，数据集会自动下载。
   2. 无网络情况下，用户需自行获取fill50k数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径，以及需要修改里面fill50k.py文件。

   ```shell
   examples/SDXL/sdxl_text2img_controlnet_deepspeed.sh
   ```
   > **注意：** 
   >需要修改数据集下面的fill50k.py文件中的57到59行，修改示例如下:
   > ```python
   > metadata_path = "数据集路径/fill50k/train.jsonl"
   > images_dir = "数据集路径/fill50k"
   > conditioning_images_dir = "数据集路径/fill50k"
   >```
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

##### 全参微调 

   > **说明：** 
   >数据集同Lora微调，请参考Lora章节。
#### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   stabilityai/stable-diffusion-xl-base-1.0 #预训练模型
   madebyollin/sdxl-vae-fp16-fix #vae模型
   ```

3. 获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径。
   ```shell
   examples/SDXL/sdxl_text2img_finetune_deepspeed.sh
   ```
#### 开始训练
1. 进入微调脚本目录

    ```
   cd diffusers/examples/SDXL
   ```

2. 运行训练的脚本。
- 单机八卡微调
  ```shell
  bash sdxl_text2img_controlnet_deepspeed.sh      #8卡deepspeed训练 sdxl_controlnet fp16
  bash sdxl_text2img_lora_deepspeed.sh            #8卡deepspeed训练 sdxl_lora fp16
  bash sdxl_text2img_finetune_deepspeed.sh        #8卡deepspeed训练 sdxl_finetune fp16
  ```

 ```
### 性能
| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| GPU | 8p |    LoRA    | 28.07 |     7      | fp16 | 2.1 | ✔ |
| Atlas A2 |8p |    LoRA    | 31.74 |     7      | fp16 | 2.1 | ✔ |
| GPU | 8p | Controlnet | 30.38  |     5      | fp16 | 2.1 | ✔ |
| Atlas A2 |8p | Controlnet | 32.43 |     5      | fp16 | 2.1 | ✔ |
| GPU | 8p |  Finetune  | 167.88 |     24     | fp16 | 2.1 | ✔ |
| Atlas A2 |8p |  Finetune  | 164.66 |     24     | fp16 | 2.1 | ✔ |

## 推理
### 环境搭建
   同微调对应章节
### 开始推理
1. 进入微调脚本目录

    ```
   cd diffusers/examples/SDXL
   ```
2、运行推理的脚本。

- 单机单卡推理
- 推理前加载环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

- 调用推理脚本
  ```shell
  python sdxl_text2img_lora_infer.py        # 混精fp16 文生图lora微调任务推理
  python sdxl_text2img_controlnet_infer.py  # 混精fp16 文生图controlnet微调任务推理
  python sdxl_text2img_infer.py             # 混精fp16 文生图全参微调任务推理
  python sdxl_img2img_infer.py              # 混精fp16 图生图微调任务推理
  ```

### 性能
| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:---:|:---:|:---:|
| GPU | 8p |    文生图lora    | 2.65 |  fp16 | 2.1 | ✔ |
| Atlas A2 |8p |    文生图lora    | 1.45 |  fp16 | 2.1 | ✔ |
| GPU | 8p | 文生图controlnet | 2.33  |  fp16 | 2.1 | ✔ |
| Atlas A2 |8p | 文生图controlnet | 1.41 |  fp16 | 2.1 | ✔ |
| GPU | 8p |  文生图全参  | 3.04 | fp16 | 2.1 | ✔ |
| Atlas A2 |8p |  文生图全参  | 1.55 | fp16 | 2.1 | ✔ |
| GPU | 8p |  图生图  | 3.02 | fp16 | 2.1 | ✔ |
| Atlas A2 |8p |  图生图  | 3.56 | fp16 | 2.1 | ✔ |


## 引用

### 公网地址说明

[代码涉及公网地址](/MindSpeed-MM/docs/public_address_statement.md)参考 docs/public_address_statement.md

