中文版本Readme请参考[README_cn.md](README_cn.md)

Open-Sora Plan v1.5.0 is trained using the MindSpeed-MM toolkit, which is licensed under the Apache License 2.0. See https://www.apache.org/licenses/LICENSE-2.0 for more details.

### Prerequisites

Open-Sora Plan v1.5.0 is trained using CANN version 8.0.1. Please refer to the official guide [CANN8_0_1](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/264595017?idAbsPath=fixnode01|23710424|251366513|22892968|252309113|251168373) for installation instructions.

Clone code:

```
git clone -b mindspeed_mmdit https://github.com/PKU-YuanGroup/Open-Sora-Plan.git
cd Open-Sora Plan
```

### Runtime Environment

1、To begin, install **Torch** and **MindSpeed** as required for the training environment.

```python
# python3.8
conda create -n osp python=3.8
conda activate osp

# Install torch and torch_npu, making sure to select the versions compatible with your Python version and system architecture (x86 or ARM), including the corresponding apex package.
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl

# apex for Ascend, refer to https://gitee.com/ascend/apex
# It is recommended to build and install from the official source repository.

# Modify the environment variable paths in the shell script to the actual paths. Example:
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# install mindspeed
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 59b4e983b7dc1f537f8c6b97a57e54f0316fafb0
pip install -r requirements.txt
pip3 install -e .
cd ..

# install other repos
pip install -e .
```

2、install decord

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

### Download Weights

Modelers:

https://modelers.cn/models/PKU-YUAN-Group/Open-Sora-Plan-v1.5.0

huggingface：

https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0

T5:

[google/t5-v1_1-xl · Hugging Face](https://huggingface.co/google/t5-v1_1-xl)

CLIP:

[laion/CLIP-ViT-bigG-14-laion2B-39B-b160k · Hugging Face](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)

### Train Text-to-Video

Make sure to properly configure `data.json` and `model_opensoraplan1_5.json`.

#### data.json: 

```
{
	"dataset_param": {
		"dataset_type": "t2v",
		"basic_parameters": {
			"data_path": "./examples/opensoraplan1.5/data.txt",
			"data_folder": "",
			"data_storage_mode": "combine"
		},
		"preprocess_parameters": {
			"video_reader_type": "decoder",
			"image_reader_type": "Image",
			"num_frames": 121, 
			"frame_interval": 1,
			"max_height": 576, # Sample height when fixed resolution is enabled; this setting is ignored when multi-resolution is enabled.
			"max_width": 1024, # Sample width when fixed resolution is enabled; this setting is ignored when multi-resolution is enabled.
			"max_hxw": 589824, # Maximum number of tokens when multi-resolution is enabled.
			"min_hxw": 589824, # Minimum number of tokens when multi-resolution is enabled. Additionally, when force_resolution is enabled, min_hxw should be set to max_height * max_width to filter out low-resolution samples, or to a custom value for stricter filtering criteria.
			"force_resolution": true, # Enable fixed-resolution training.
			"force_5_ratio": false, # Enable multi-resolution training with 5 aspect ratios.
			"max_h_div_w_ratio": 1.0, # Maximum allowed aspect ratio for filtering.
			"min_h_div_w_ratio": 0.42, # Minimum allowed aspect ratio for filtering.
			"hw_stride": 16,
			"ae_stride_t": 8,
			"train_fps": 24, # Sampling FPS during training; all videos with varying frame rates will be resampled to train_fps.
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
        "sampler_type": "LengthGroupedSampler", # Enable the Group Data strategy (enabled by default).
		"batch_size": 1,
		"num_workers": 4,
		"shuffle": false,
		"drop_last": true,
		"pin_memory": false,
		"group_data": true,
		"initial_global_step_for_sampler": 0, 
		"gradient_accumulation_size": 4,
		"collate_param": {
			"model_name": "GroupLength", # Enable the Group Data-specific collate function (enabled by default).
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
    "enable_encoder_dp": true, # MindSpeed optimization. It takes effect when TP (tensor parallelism) degree is greater than 1.
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
        "use_tiling": true, # Whether to enable the tiling strategy.
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
        "num_layers": [2, 4, 6, 8, 6, 4, 2], # Number of layers per stage.
        "sparse_n": [1, 2, 4, 8, 4, 2, 1], # Sparsity level for each stage.
        "double_ff": true, # Whether to use a shared FFN for visual and textual inputs, or separate FFNs for each.
        "sparse1d": true, # Whether to use the Skiparse strategy; setting this to false results in a dense DiT.
        "num_heads": 24,
        "head_dim": 128,
        "in_channels": 32,
        "out_channels": 32,
        "timestep_embed_dim": 1024,
        "caption_channels": 2048,
        "pooled_projection_dim": 1280,
        "skip_connection": true, # Whether to add skip connections.
        "dropout": 0.0, 
        "attention_bias": true,
        "patch_size": 2,
        "patch_size_t": 1,
        "activation_fn": "gelu-approximate",
        "norm_elementwise_affine": false,
        "norm_eps": 1e-06,
        "from_pretrained": null # Path to the pretrained weights; merged weights must be used.
    },
    "diffusion": {
        "model_id": "OpenSoraPlan",
        "weighting_scheme": "logit_normal",
        "use_dynamic_shifting": true 
    }
}

```

Enter the Open-Sora Plan directory and run:

```
bash examples/opensoraplan1.5/pretrain_opensoraplan1_5.sh
```

**Parameter Description:**

`--optimizer-selection fused_ema_adamw`  Select the optimizer to use. In our case, `fused_ema_adamw` is required to obtain EMA-based weights.

`--model_custom_precision`  Different components use different precisions, rather than adopting Megatron’s default of full-model bf16 precision. For example, the VAE is run in fp32, while the text encoder and DiT use bf16.

`--clip_grad_ema_decay 0.99` Set the EMA decay rate used in adaptive gradient clipping.

`--selective_recom`  `--recom_ffn_layers 32`  Whether to enable selective recomputation and specify the number of layers for it. When selective recomputation is activated, only the FFN layers are recomputed, while the Attention layers are skipped, enabling faster training. This parameter is mutually exclusive with `--recompute-granularity full`, `--recompute-method block`, and `--recompute-num-layers 0`. When selective recomputation is enabled, full-layer recomputation is disabled by default.

### Sample Text-to-Video

Due to TP-based training, the model weights are partitioned. Therefore, weight merging is required prior to running inference.

#### Merge Weights

```
python examples/opensoraplan1.5/convert_mm_to_ckpt.py --load_dir $load_dir --save_dir $save_dir --ema
```

**Parameter Description:**

`--load_dir`: Path to the weights saved during training, partitioned by Megatron.

`--save_dir`: Path to save the merged weights.

`--ema`: Whether to use EMA (Exponential Moving Average) weights.

#### Inference

Make sure the `inference_t2v_model1_5.json` file is properly configured.

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
        "use_tiling": true, # Whether to enable the tiling strategy; it is enabled by default during inference to reduce memory usage.
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
        "num_inference_steps": 50, # Inference steps
        "guidance_scale": 8.0, # CFG strength. We recommend using a relatively high value; 8.0 is generally a good choice.
        "guidance_rescale": 0.7, # Guidance rescale strength. If the sampled outputs appear overly saturated, we recommend increasing guidance_rescale instead of adjusting the CFG value.
        "use_linear_quadratic_schedule": false, # Using a linear-to-quadratic sampling strategy.
        "use_dynamic_shifting": false,
        "shift": 7.0 # Using the shifting sampling strategy.
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

Enter the Open-Sora Plan directory and run:

```
bash examples/opensoraplan1.5/inference_t2v_1_5.sh
```

In practice, inference at 121×576×1024 resolution can be run with TP=1 (i.e., without parallelism). To accelerate inference, you may manually increase the TP parallelism level.
