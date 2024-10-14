# LLaVA1.5 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换](#jump2.2)
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

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
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
**注意事项:** 

  需要修改 mindspeed/core/transformer/dot_product_attention.py的65行，修改如下：
```
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

- [ViT-L-14-336px](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)：CLIPViT模型；

- [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/)： GPT模型；

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeeed-MM修改了部分原始网络的结构名称，因此需要使用如下脚本代码对下载的预训练权重进行转换。 当前训练只使用了ViT-L-14-336px和lmsys/vicuna-7b-v1.5两个模型，以下介绍这两个模型从开源仓转换成MindSpeeed-MM所需权重的方法：

- ViT-L-14-336px权重转换

  参考 NVIDIA/Megatron-LM中[Vision model](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/multimodal/README.md#vision-model) , 
  执行如下命令：
  ```
  python examples/multimodal/clip_converter.py --download-root /some/download/folder --output /some/output/folder --tensor-parallel-size 1 --use-te
  ```
  如果执行环境连接不到外网下载ViT-L-14-336px模型，建议手动下载，再在clip_converter.py中将ViT-L-14-336px路径修改成本地路径
  ```
  model, _ = clip.load("{dir_to_model}/ViT-L-14-336px.pt", device=device, download_root="")
  ```
  其中{dir_to_model}为模型所在的路径。 
  转换的结果在： /some/output/folder/iter_0000001/mp_rank_00/model_optim_rng.pt
  
  对于转换后的结果，需要再执行如下转换，其中{target_dir}为最终的权重文件保存路径：
  ```
  before = torch.load("/some/output/folder/iter_0000001/mp_rank_00/model_optim_rng.pt")["model"]
  torch.save(before, "{target_dir}/final_vit_pt_file.pt")
  ```

- lmsys/vicuna-7b-v1.5权重转换

  参考[ModelLink](https://gitee.com/ascend/ModelLink/blob/master/examples/README.md#21-huggingface%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2%E5%88%B0megatron-lm%E6%A0%BC%E5%BC%8F)中语言模型权重转换的脚本：
  ```
  source {cann_dir}/ascend-toolkit/set_env.sh
  HF_FORMAT_DIR="{dir_to_model}/vicuna-7b-v1.5"
  MEGATRON_FORMAT_DIR="{target_dir}"
  TOKENIZER_MODEL="{dir_to_model}/vicuna-7b-v1.5/tokenizer.model" 
  python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader llama2_hf \
       --saver megatron \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --load-dir ${HF_FORMAT_DIR} \
       --save-dir ${MEGATRON_FORMAT_DIR} \
       --tokenizer-model ${TOKENIZER_MODEL} \
       --params-dtype bf16
  ```
  其中： {dir_to_model}为vicuna-7b-v1.5所在路径，{target_dir}为转换结果文件路径, {cann_dir}为cann包安装路径。转换的结果在：{target_dir}/iter_0000001/mp_rank_00/model_optim_rng.pt。

由于MindSpeed-MM中模型变量名称跟转换结果有差异，需要再做一次适配：
  - 在megatron同级目录，创建convert.py脚本，将如下代码复制到convert.py中，
  - 修改{target_dir}为上一步model_optim_rng.pt所在路径，
  - 修改{dir_to_save_file}为结果文件所在路径，
  - 执行命令：python convert.py
  ```
  import torch
  def convert_param():
      ckp = torch.load("{target_dir}/model_optim_rng.pt")["model"]["language_model"]
      target_ckp = {}
      target_ckp["embedding.word_embeddings.weight"] = ckp["embedding"]["word_embeddings"]["weight"]
      target_ckp["output_layer.weight"] = ckp["output_layer"]["weight"]
      for encode_key in ckp["encoder"].keys():
          if ckp["encoder"][encode_key] is not None:
              targetkey = encode_key.replace("input_norm", "input_layernorm")
              targetkey = targetkey.replace(".dense.", ".linear_proj.")
              targetkey = targetkey.replace("query_key_value", "linear_qkv")
              targetkey = targetkey.replace("post_attention_norm", "pre_mlp_layernorm")
              targetkey = targetkey.replace("dense_h_to_4h", "linear_fc1")
              targetkey = targetkey.replace("dense_4h_to_h", "linear_fc2")
              targetkey = targetkey.replace("final_norm", "final_layernorm")
              targetkey = "decoder." + targetkey
              target_ckp[targetkey] = ckp["encoder"][encode_key]
      torch.save(target_ckp, "{dir_to_save_file}/xxx.pt")

  if __name__ == "__main__":
      convert_param()
  ```
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
