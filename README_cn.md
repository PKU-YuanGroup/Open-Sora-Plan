Open-Sora Plan v1.5.0采用mindspeed-mm套件训练。mindspeed-mm套件采用apache 2.0协议，请访问 https://www.apache.org/licenses/LICENSE-2.0 查看细节。

### 前置要求

Open-Sora Plan v1.5.0在CANN 8.0.1版本完成训练，请参照[CANN 系列 昇腾计算 8.0.1 软件补丁下载](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/264595017?idAbsPath=fixnode01|23710424|251366513|22892968|252309113|251168373)安装。

仓库克隆：

```
git clone -b mindspeed_mmdit https://github.com/PKU-YuanGroup/Open-Sora-Plan.git
cd Open-Sora-Plan
```

### 环境安装

1、安装torch、Mindspeed

```python
# python3.8
conda create -n osp python=3.8
conda activate osp

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl

# apex for Ascend 参考 https://gitee.com/ascend/apex
# 建议从原仓编译安装

# 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 59b4e983b7dc1f537f8c6b97a57e54f0316fafb0
pip install -r requirements.txt
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -e .
```

2、安装decord

```bash
git clone --recursive https://github.com/dmlc/decord
mkdir build && cd build 
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/usr/local/ffmpeg 
make 
cd ../python 
pwd=$PWD 
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc 
source ~/.bashrc 
python3 setup.py install --user
```

### 权重下载

魔乐社区：

https://modelers.cn/models/PKU-YUAN-Group/Open-Sora-Plan-v1.5.0

huggingface：

https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0

T5:

[google/t5-v1_1-xl · Hugging Face](https://huggingface.co/google/t5-v1_1-xl)

CLIP:

[laion/CLIP-ViT-bigG-14-laion2B-39B-b160k · Hugging Face](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)

### Train Text-to-Video

需要设置好data.json和model_opensoraplan1_5.json。

#### data.json: 

```
{
	"dataset_param": {
		"dataset_type": "t2v",
		"basic_parameters": {
			"data_path": "./examples/opensoraplan1.5/data.txt", # 数据路径
			"data_folder": "",
			"data_storage_mode": "combine"
		},
		"preprocess_parameters": {
			"video_reader_type": "decoder",
			"image_reader_type": "Image",
			"num_frames": 121, 
			"frame_interval": 1,
			"max_height": 576, # 开启固定分辨率时的样本高度，在开启多分辨率时无效
			"max_width": 1024, # 开启固定分辨率时的样本宽度，在开启多分辨率时无效
			"max_hxw": 589824, # 开启多分辨率时的最大token数
			"min_hxw": 589824, # 开启多分辨率时的最小token数。此外，min_hxw需要在开启force_resolution时设置为max_height * max_width以过滤低分辨率样本，或自定义更严格的筛选标准
			"force_resolution": true, # 开启固定分辨率训练
			"force_5_ratio": false, # 开启5宽高比多分辨率策略训练
			"max_h_div_w_ratio": 1.0, # 筛选最大高宽比
			"min_h_div_w_ratio": 0.42, # 筛选最小高宽比
			"hw_stride": 16,
			"ae_stride_t": 8,
			"train_fps": 24, # 训练时采样fps，会将不同fps的视频都重采样到train_fps
			"speed_factor": 1.0,
			"drop_short_ratio": 1.0,
			"min_num_frames": 29,
			"cfg": 0.1,
			"batch_size": 1,
			"gradient_accumulation_size": 4,
			"use_aesthetic": false,
			"train_pipeline": {
				"video": [{
						"trans_type": "ToTensorVideo"
					},
					{
						"trans_type": "CenterCropResizeVideo",
						"param": {
							"size": [576, 1024],
							"interpolation_mode": "bicubic"
						}
					},
					{
						"trans_type": "ae_norm"
					}
				],
				"image": [{
					"trans_type": "ToTensorVideo"
					},
					{
						"trans_type": "CenterCropResizeVideo",
						"param": {
							"size": [576, 1024],
							"interpolation_mode": "bicubic"
						}
					},
					{
						"trans_type": "ae_norm"
					}
				]
			}
		},
		"use_text_processer": true,
		"enable_text_preprocess": true,
		"model_max_length": 512,
		"tokenizer_config": {
			"hub_backend": "hf",
			"autotokenizer_name": "AutoTokenizer",
			"from_pretrained": "/work/share/checkpoint/pretrained/t5/t5-v1_1-xl"
		},
		"tokenizer_config_2": {
		    "hub_backend": "hf",
			"autotokenizer_name": "AutoTokenizer",
			"from_pretrained": "/work/share/checkpoint/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189"
		},
		"use_feature_data": false,
		"use_img_from_vid": false
	},
	"dataloader_param": {
		"dataloader_mode": "sampler",
        "sampler_type": "LengthGroupedSampler", # 开启Group Data策略，默认指定
		"batch_size": 1,
		"num_workers": 4,
		"shuffle": false,
		"drop_last": true,
		"pin_memory": false,
		"group_data": true,
		"initial_global_step_for_sampler": 0, 
		"gradient_accumulation_size": 4,
		"collate_param": {
			"model_name": "GroupLength", # 开启Group Data对应的Collate，默认指定
			"batch_size": 1,
			"num_frames": 121,
			"group_data": true,
			"ae_stride": 8,
			"ae_stride_t": 8,
			"patch_size": 2,
			"patch_size_t": 1
		}
	}
}

```

#### model_opensoraplan1_5.json

```
{
    "frames": 121,
    "allow_tf32": false,
    "allow_internal_format": false,
    "load_video_features": false,
    "load_text_features": false,
    "enable_encoder_dp": true, # mindspeed架构优化，在TP并行度大于1时起作用
    "weight_dtype": "bf16",
    "ae": {
        "model_id": "wfvae",
        "base_channels": 160,
        "connect_res_layer_num": 1,
        "decoder_energy_flow_hidden_size": 128,
        "decoder_num_resblocks": 2,
        "dropout": 0.0,
        "encoder_energy_flow_hidden_size": 128,
        "encoder_num_resblocks": 2,
        "l1_dowmsample_block": "Spatial2xTime2x3DDownsample",
        "l1_downsample_wavelet": "HaarWaveletTransform3D",
        "l1_upsample_block": "Spatial2xTime2x3DUpsample",
        "l1_upsample_wavelet": "InverseHaarWaveletTransform3D",
        "l2_dowmsample_block": "Spatial2xTime2x3DDownsample",
        "l2_downsample_wavelet": "HaarWaveletTransform3D",
        "l2_upsample_block": "Spatial2xTime2x3DUpsample",
        "l2_upsample_wavelet": "InverseHaarWaveletTransform3D",
        "latent_dim": 32,
        "norm_type": "layernorm",
        "scale": [0.7031, 0.7109, 1.5391, 1.2969, 0.7109, 1.4141, 1.3828, 2.1719, 1.7266,
        1.8281, 1.9141, 1.2031, 0.6875, 0.9609, 1.6484, 1.1875, 1.5312, 1.1328,
        0.8828, 0.6836, 0.8828, 0.9219, 1.6953, 1.4453, 1.5312, 0.6836, 0.7656,
        0.8242, 1.2344, 1.0312, 1.7266, 0.9492],
        "shift": [-0.2129,  0.1226,  1.6328,  0.6211, -0.8750,  0.6172, -0.5703,  0.1348,
        -0.2178, -0.9375,  0.3184,  0.3281, -0.0544, -0.1826, -0.2812,  0.4355,
         0.1621, -0.2578,  0.7148, -0.7422, -0.2295, -0.2324, -1.4922,  0.6328,
         1.1250, -0.2578, -2.1094,  1.0391,  1.1797, -1.2422, -0.2988, -0.9570],
        "t_interpolation": "trilinear",
        "use_attention": true,
        "use_tiling": true, # 是否开启tiling策略
        "from_pretrained": "/work/share/checkpoint/pretrained/vae/Middle888/merged.ckpt",
        "dtype": "fp32"
      },
    "text_encoder": {
        "hub_backend": "hf",
        "model_id": "T5",
        "from_pretrained": "/work/share/checkpoint/pretrained/t5/t5-v1_1-xl",
        "low_cpu_mem_usage": false
    },
    "text_encoder_2":{
        "hub_backend": "hf",
        "model_id": "CLIPWithProjection", 
        "from_pretrained": "/work/share/checkpoint/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189",
        "low_cpu_mem_usage": false
    },
    "predictor": {
        "model_id": "SparseUMMDiT",
        "num_layers": [2, 4, 6, 8, 6, 4, 2], # 每个stage的层数
        "sparse_n": [1, 2, 4, 8, 4, 2, 1], # 每个stage的稀疏度
        "double_ff": true, # 采用visual和text共享FFN还是各自独立FFN
        "sparse1d": true, # 是否采用Skiparse策略，设置为false则为dense dit
        "num_heads": 24,
        "head_dim": 128,
        "in_channels": 32,
        "out_channels": 32,
        "timestep_embed_dim": 1024,
        "caption_channels": 2048,
        "pooled_projection_dim": 1280,
        "skip_connection": true, # 是否添加skip connection
        "dropout": 0.0, 
        "attention_bias": true,
        "patch_size": 2,
        "patch_size_t": 1,
        "activation_fn": "gelu-approximate",
        "norm_elementwise_affine": false,
        "norm_eps": 1e-06,
        "from_pretrained": null # 预训练权重路径，需采用合并后的权重
    },
    "diffusion": {
        "model_id": "OpenSoraPlan",
        "weighting_scheme": "logit_normal",
        "use_dynamic_shifting": true 
    }
}

```

进入Open-Sora Plan目录下，运行

```
bash examples/opensoraplan1.5/pretrain_opensoraplan1_5.sh
```

参数解析：

`--optimizer-selection fused_ema_adamw` 选择使用的优化器，我们这里需要选择fused_ema_adamw以获得EMA版本权重。

`--model_custom_precision` 不同组件使用不同的精度，而不是采用megatron默认的整网bf16精度。例如对VAE使用fp32精度，对text encoder、dit使用bf16精度。

`--clip_grad_ema_decay 0.99` 设置adaptive grad clipping中使用的EMA衰减率。

`--selective_recom`  `--recom_ffn_layers 32` 是否开启选择性重加算及选择性重计算的层数。在开启选择性重计算时，我们只对FFN进行重计算而不对Attention进行重计算，以获得加速训练效果。该参数与`--recompute-granularity full` `--recompute-method block` `--recompute-num-layers 0` 互斥，当开启选择性重计算时，默认全重计算已关闭。

### Sample Text-to-Video

由于模型训练时进行了TP切分，所以我们需要先将切分后的权重进行合并，然后再进行推理。

#### 合并权重

```
python examples/opensoraplan1.5/convert_mm_to_ckpt.py --load_dir $load_dir --save_dir $save_dir --ema
```

参数解析：

`--load_dir` 训练时经过megatron切分后保存的权重路径

`--save_dir` 合并后的权重路径

`--ema` 是否采用EMA权重

### 推理

需要配置好inference_t2v_model1_5.json。

```
{
    "ae": {
        "model_id": "wfvae",
        "base_channels": 160,
        "connect_res_layer_num": 1,
        "decoder_energy_flow_hidden_size": 128,
        "decoder_num_resblocks": 2,
        "dropout": 0.0,
        "encoder_energy_flow_hidden_size": 128,
        "encoder_num_resblocks": 2,
        "l1_dowmsample_block": "Spatial2xTime2x3DDownsample",
        "l1_downsample_wavelet": "HaarWaveletTransform3D",
        "l1_upsample_block": "Spatial2xTime2x3DUpsample",
        "l1_upsample_wavelet": "InverseHaarWaveletTransform3D",
        "l2_dowmsample_block": "Spatial2xTime2x3DDownsample",
        "l2_downsample_wavelet": "HaarWaveletTransform3D",
        "l2_upsample_block": "Spatial2xTime2x3DUpsample",
        "l2_upsample_wavelet": "InverseHaarWaveletTransform3D",
        "latent_dim": 32,
        "vae_scale_factor": [8, 8, 8],
        "norm_type": "layernorm",
        "scale": [0.7031, 0.7109, 1.5391, 1.2969, 0.7109, 1.4141, 1.3828, 2.1719, 1.7266,
        1.8281, 1.9141, 1.2031, 0.6875, 0.9609, 1.6484, 1.1875, 1.5312, 1.1328,
        0.8828, 0.6836, 0.8828, 0.9219, 1.6953, 1.4453, 1.5312, 0.6836, 0.7656,
        0.8242, 1.2344, 1.0312, 1.7266, 0.9492],
        "shift": [-0.2129,  0.1226,  1.6328,  0.6211, -0.8750,  0.6172, -0.5703,  0.1348,
        -0.2178, -0.9375,  0.3184,  0.3281, -0.0544, -0.1826, -0.2812,  0.4355,
         0.1621, -0.2578,  0.7148, -0.7422, -0.2295, -0.2324, -1.4922,  0.6328,
         1.1250, -0.2578, -2.1094,  1.0391,  1.1797, -1.2422, -0.2988, -0.9570],
        "t_interpolation": "trilinear",
        "use_attention": true,
        "use_tiling": true, # 是否开启tiling策略，推理时默认开启节省显存
        "from_pretrained": "/work/share/checkpoint/pretrained/vae/Middle888/merged.ckpt",
        "dtype": "fp16"
      },
    "text_encoder": {
        "hub_backend": "hf",
        "model_id": "T5",
        "from_pretrained": "/work/share/checkpoint/pretrained/t5/t5-v1_1-xl",
        "low_cpu_mem_usage": false
    },
    "text_encoder_2":{
        "hub_backend": "hf",
        "model_id": "CLIPWithProjection", 
        "from_pretrained": "/work/share/checkpoint/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189",
        "low_cpu_mem_usage": false
    },
    "tokenizer":{
        "hub_backend": "hf",
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "/work/share/checkpoint/pretrained/t5/t5-v1_1-xl",
        "low_cpu_mem_usage": false
    },
    "tokenizer_2":{
        "hub_backend": "hf",
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "/work/share/checkpoint/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189",
        "low_cpu_mem_usage": false
    },
    "predictor": {
        "model_id": "SparseUMMDiT",
        "num_layers": [2, 4, 6, 8, 6, 4, 2],
        "sparse_n": [1, 2, 4, 8, 4, 2, 1],
        "double_ff": true,
        "sparse1d": true,
        "num_heads": 24,
        "head_dim": 128,
        "in_channels": 32,
        "out_channels": 32,
        "timestep_embed_dim": 1024,
        "caption_channels": 2048,
        "pooled_projection_dim": 1280,
        "skip_connection": true,
        "skip_connection_zero_init": true,
        "dropout": 0.0,
        "attention_bias": true,
        "patch_size": 2,
        "patch_size_t": 1,
        "activation_fn": "gelu-approximate",
        "norm_elementwise_affine": true,
        "norm_eps": 1e-06,
        "from_pretrained": "/path/to/pretrained/model"
    },
    "diffusion": {
        "model_id": "OpenSoraPlan",
        "num_inference_steps": 50, # 推理步数
        "guidance_scale": 8.0, # CFG强度，我们推荐较大的CFG，8.0是较好的值
        "guidance_rescale": 0.7, # guidance rescale强度，如认为采样饱和度过高，我们推荐将gudance_rescale增大，而非调整CFG
        "use_linear_quadratic_schedule": false, # 采用线性——平方采样策略
        "use_dynamic_shifting": false,
        "shift": 7.0 # 采用shifting采样策略
    },
    "pipeline_config": {
        "use_attention_mask": true,
        "input_size": [121, 576, 1024],
        "version": "v1.5",
        "model_type": "t2v"
    },
    "micro_batch_size": 1,
    "frame_interval":1,
    "model_max_length": 512,
    "save_path":"./opensoraplan_samples/test_samples",
    "fps":24,
    "prompt":"./examples/opensoraplan1.5/sora.txt",
    "device":"npu",
    "weight_dtype": "fp16"
}

```

进入Open-Sora Plan目录下，运行

```
bash examples/opensoraplan1.5/inference_t2v_1_5.sh
```

实测TP=1即不开启并行策略能够运行121x576x1024推理，如需加快推理速度请自行调节TP并行度。
