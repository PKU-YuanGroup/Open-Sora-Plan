
### Data prepare
We use a `data.txt` file to specify all the training data. Each line in the file consists of `DATA_ROOT` and `DATA_JSON`. The example of `data.txt` is as follows. The `.pkl` also be supported, because `.pkl` can save memory and speed up.
```
/path/to/data_root_1,/path/to/data_json_1.json
/path/to/data_root_2,/path/to/data_json_2.json
...
```
Then, we introduce the format of the annotation json file. The absolute data path is the concatenation of `DATA_ROOT` and the `"path"` field in the annotation json file.

#### For image
The format of image annotation file is as follows.
```
[
  {
    "path": "00168/001680102.jpg",
    "cap": [
      "a image caption."
    ],
    "resolution": {
      "height": 512,
      "width": 683
    }, 
    "aesthetic": 5.3, 
  },
  ...
]
```

#### For video
The format of video annotation file is as follows. 

The keys "aesthetic," "cut," "crop," and "tech" can all be optional. If "aesthetic" is missing, no additional high-aesthetic prompt will be applied. If "cut" is missing, the entire video frame is assumed to be usable. If "crop" is missing, the full resolution is assumed to be usable. "Tech" and "motion" are used only for preprocessing and filtering. For more details, please refer to our [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.3.0.md#data-construction).

```
[
  {
    "path": "panda70m_part_5565/qLqjjDhhD5Q/qLqjjDhhD5Q_segment_0.mp4",
    "cap": [
      "A man and a woman are sitting down on a news anchor talking to each other."
    ],
    "resolution": {
      "height": 720,
      "width": 1280
    },
    "num_frames": 100, 
    "fps": 24, 
    "aesthetic": 5.3, 
    "cut": [start_frame_idx, end_frame_idx],
    "crop": [start_of_x, end_of_x, start_of_y, end_of_y], 
    "tech": 1.1, 
    "motion": 0.02
  },
  ...
]
```

### Training
```
bash scripts/text_condition/gpu/train_t2v.sh
```

We introduce some key parameters in order to customize your training process.

| Argparse | Usage |
|:---|:---|
|_Training size_||
|`--num_frames 61`|To train videos of different durations, e.g, 29, 61, 93, 125...|
|`--drop_short_ratio 1.0`|Do not want to train on videos of dynamic durations, discard all video data with frame counts not equal to `--num_frames`|
|`--max_height 640`|To train videos of different resolutions|
|`--max_width 480`|To train videos of different resolutions|
|`--force_resolution`| Fixed resolution training: If enabled, the training resolution is determined by the specified `--max_height` and `--max_width`.|
|`--max_hxw`| The product of the maximum resolution.  |
|`--min_hxw`|The product of the minimum resolution. |
|_Data processing_||
|`--data /path/to/data.txt`|Specify your training data.|
|`--speed_factor 1.25`|To accelerate 1.25x videos. |
|`--group_data`|If you want to train with videos of dynamic durations, we highly recommend specifying `--group_data` as well. It improves computational efficiency during training.|
|`--hw_stride`|Minimum step of resolution|
|_Load weights_||
|`--pretrained`|This is typically used for loading pretrained weights across stages, such as using 240p weights to initialize 480p training. Or when switching datasets and you do not want the previous optimizer state.|
|`--resume_from_checkpoint`|It will resume the training process from the latest checkpoint in `--output_dir`. Typically, we set `--resume_from_checkpoint="latest"`, which is useful in cases of unexpected interruptions during training.|
|_Sequence Parallelism_||
|`--sp_size 8 --train_sp_batch_size 2`|It means running a batch size of 2 across 8 GPUs (8 GPUs on the same node).|

> [!Warning]
> <div align="left">
> <b>
> 🚨 We have two ways to load weights: `--pretrained` and `--resume_from_checkpoint`. The latter will override the former.
> </b>
> </div>

To prevent confusion, we present some examples below:

Case 1: If you want a fixed resolution of 480P and a fixed frame count of 93 frames. 

```
--max_height 480 --max_width 640 --force_resolution \
--num_frames 93 --drop_short_ratio 1.0
```

Case 2: If you want to enable variable duration.

```
--drop_short_ratio 0.0 --group_data
```

Case 3: If you want to enable variable resolution, there are two approaches: one is using an absolute resolution, and the other is based on the resolution's area.

```
# absolute resolution
--max_height 480 --max_width 640 --min_height 320 --min_width 320 --group_data
# resolution's area
--max_hxw 262144 --min_hxw 65536 --group_data
```

Finally, we can combine these approaches to enable bucketed training with variable resolution and variable duration.

```
--max_hxw 262144 --min_hxw 65536 --group_data \
--num_frames 93 --drop_short_ratio 0.0
```

### Inference

You need download the models manually.
First, you need to download checkpoint including [diffusion model](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/any93x640x640), [vae](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/vae) and [text encoder](https://huggingface.co/google/mt5-xxl). The [prompt refiner](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/prompt_refiner) is optional.
Then, modify `--model_path`, `--text_encoder_name_1` and `--ae_path` the path in [script](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/scripts/text_condition/gpu/sample_t2v_v1_3.sh#L4). The `--caption_refiner` is optional.

We provide multiple inference scripts to support various requirements. We recommend configuration `--guidance_scale 7.5 --num_sampling_steps 100 --sample_method EulerAncestralDiscrete` for sampling.

#### 🖥️ 1 GPU 
If you only have one GPU, it will perform inference on each sample sequentially, one at a time.

Remove `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8` to `CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node 1`.

```
bash scripts/text_condition/gpu/sample_t2v_v1_3.sh
```

#### 🖥️🖥️ Multi-GPUs 
If you want to batch infer a large number of samples, each GPU will infer one sample.

Remember add `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8`.

```
bash scripts/text_condition/gpu/sample_t2v_v1_3.sh
```

#### 🖥️🖥️ Multi-GPUs & Sequence Parallelism 
If you want to quickly infer one sample, it will utilize all GPUs simultaneously to infer that sample.

You only add `--sp` to enable sequence parallelism inferencing, but it should based on `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8`.
```
bash scripts/text_condition/gpu/sample_t2v_sp.sh
```