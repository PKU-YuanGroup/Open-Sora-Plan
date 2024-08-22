# Diffusers

<p align="left">
        <b>简体中文</b> |
</p>

- [SD3](#StableDiffusionXL)
  - [模型介绍](#模型介绍)
  - [预训练](#预训练) 
    - [环境搭建](#环境搭建)
    - [性能](#性能)
      - [吞吐](#吞吐)
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

## 预训练

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

    3.1 【下载 SD3 [GitHub参考实现](https://github.com/huggingface/diffusers) 或 [适配昇腾AI处理器的实现](https://gitee.com/ascend/ModelZoo-PyTorch.git) 或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    code_path=examples/dreambooth/
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
    
    3.2【安装其余依赖库】

    ```shell
    pip install -r requirements.txt
    pip install -r examples/dreambooth/requirements_sd3.txt # 安装对应依赖
    ```

4. 预训练

    4.1 【准备预训练数据集】

    用户需自行获取并解压LAION_5B数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    ```shell
    dataset_name="laion5b" # 数据集 路径
    ```
    只包含图片的训练数据集，如非deepspeed脚本使用训练数据集dog(下载地址：https://huggingface.co/datasets/diffusers/dog-example)，在shell启动脚本中将`input_dir`参数设置为本地数据集绝对路径，
    ```shell
    input_dir="dog" # 数据集路径
    ```

    4.2 【配置 SD3 预训练脚本】

    联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载

    获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径

    ```shell
    scripts_path="./sd3" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型
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

    4.4 【启动 SD3 预训练脚本】

    本任务主要提供**混精fp16**和**混精bf16**两种**8卡**训练脚本，默认使用**deepspeed**分布式训练。

    ```shell
    train_sd3_deepspeed_**16.sh
    ```
   
### 性能

#### 吞吐

SD3 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 设备   | 模型        | 迭代数  | 样本吞吐 (f/p/s) | token吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 浮点计算数 (TFLOPs/s) |
|------|-----------|------|--------------------|----------------------|-----------------|------------------|
| NPUs | SD3_train_bf16  | * | .               | .                 | .            | .            |
| 参考 | SD3_train_bf16  | * | .               | .                 | .            | .            |
| NPUs | SD3_train_fp16 | * | .               | .                 | .            | .           |
| 参考 | SD3_train_fp16 | * | .               | .                 | .           | .           |


## 推理

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

    3.1 【下载 SD3 [GitHub参考实现](https://github.com/huggingface/diffusers) 或 [适配昇腾AI处理器的实现](https://gitee.com/ascend/ModelZoo-PyTorch.git) 或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    code_path=examples/dreambooth/
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
    
    3.2【安装其余依赖库】

    ```shell
    pip install -r requirements.txt
    pip install -r examples/dreambooth/requirements_sd3.txt # 安装对应依赖
    ```

4. 预训练

    4.1 【准备预训练数据集】

    用户需自行获取并解压LAION_5B数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    ```shell
    dataset_name="laion5b" # 数据集 路径
    ```
    只包含图片的训练数据集，如非deepspeed脚本使用训练数据集dog(下载地址：https://huggingface.co/datasets/diffusers/dog-example)，在shell启动脚本中将`input_dir`参数设置为本地数据集绝对路径，
    ```shell
    input_dir="dog" # 数据集路径
    ```

    4.2 【配置 SD3 预训练脚本】

    联网情况下，预训练模型会自动下载。无网络时，用户可访问huggingface官网自行下载

    获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径

    ```shell
    scripts_path="./sd3" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型
    dataset_name="laion5b" 
    batch_size=4
    max_train_steps=2000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="${scripts_path}/pretrain_${mixed_precision}_accelerate_config.yaml"
    ```

    4.3 【启动 SD3 预训练脚本】

    本任务主要提供**混精fp16**和**混精bf16**两种**8卡**训练脚本，默认使用**deepspeed**分布式训练。

    ```shell
    train_sd3_deepspeed_**16.sh
    ```
    4.4 【启动 SD3 推理脚本】
	1、进入推理脚本目录

   ```shell
    cd examples/SD3
   ```
    2、运行推理的脚本

    推理前加载环境变量

    ```shell
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
    调用推理脚本，图生图推理脚本需先准备图片到当前路径下（下载地址：    https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

   ```shell
   python infer_sd3_img2img_fp16.py   # 单卡推理，文生图
   python infer_sd3_text2img_fp16.py  # 单卡推理，图生图
   ```

## 性能

### 吞吐

SD3 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 | 任务 | FPS | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 竞品A | 8p | DreamBooth-LoRA | 15.04 | 8 | fp16 | 2.1 | ✘ |
| Atlas 200T A2 Box16 |8p | DreamBooth-LoRA | 14.04 | 8 | fp16 | 2.1 | ✘ |
| 竞品A | 8p | DreamBooth | 1.51 | 1 | fp16 | 2.1 | ✘ |
| Atlas 200T A2 Box16 |8p | DreamBooth | 1.34  | 1 | fp16 | 2.1 | ✘ |


## 使用基线数据集进行评估


## 引用

### 公网地址说明

代码涉及公网地址参考 public_address_statement.md

