

### Data prepare

Data preparation aligns with T2V section.

### Training

Training on GPUs:

```bash
bash scripts/text_condition/gpu/train_inpaint_v1_3.sh
```

Training on NPUs:

```bash
bash scripts/text_condition/npu/train_inpaint_v1_3.sh
```

There are additional parameters you need to understand beyond those introduced in the T2V section.

| Argparse                              | Usage                                                        |
| ------------------------------------- | ------------------------------------------------------------ |
| `--pretrained_transformer_model_path` | The function is identical to the `--pretrained` parameter in the T2V section. |
| `--default_text_ratio` 0.5            | During I2V training, a portion of the text is replaced with a default text to account for cases where the user provides an image without accompanying text. |
| `--mask_config`                       | The path of the `mask_config` file.                          |

In Open-Sora Plan V1.3, all mask ratio settings are specified in the `mask_config` file, located at `scripts/train_configs/mask_config.yaml`. The parameters include:

| Argparse                     | Usage                                                        |
| ---------------------------- | ------------------------------------------------------------ |
| `min_clear_ratio`            | The minimum ratio of frames retained during continuation and random masking. |
| `max_clear_ratio`            | The maximum ratio of frames retained during continuation and random masking. |
| `mask_type_ratio_dict_video` | During training, specify the ratio for each mask task. For video data, there are six mask types: `t2iv`, `i2v`, `transition`, `continuation`, `clear`, and `random_temporal`. These inputs will be normalized to ensure their sum equals one. |
| `mask_type_ratio_dict_image` | During training, specify the ratio for each mask task. For image data, there are two mask types: `t2iv` and `clear`. These inputs will be normalized to ensure their sum equals one. |

### Inference

Inference on GPUs:

```bash
bash scripts/text_condition/gpu/sample_inpaint_v1_3.sh
```

Inference on NPUs:

```bash
bash scripts/text_condition/npu/sample_inpaint_v1_3.sh
```

In the current version, we have only open-sourced the 93x480p version of the Image-to-Video (I2V) model. We recommend configuration `--guidance_scale 7.5 --num_sampling_steps 100 --sample_method EulerAncestralDiscrete` for sampling. 

**Inference on 93×480p**, the speed on H100 and Ascend 910B.

| Size    | 1 H100       | 1 Ascend 910B |
| ------- | ------------ | ------------- |
| 93×480p | 150s/100step | 292s/100step  |

During inference, you can specify `--nproc_per_node` and set the `--sp` parameter to choose between single-gpu/npu mode, DDP (Distributed Data Parallel) mode, or SP (Sequential Parallel) mode for inference.

The following are key parameters required for inference:

| Argparse                                       | Usage                                                        |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `--height` 352  `--width` 640  `--crop_for_hw` | When `crop_for_hw` is specified, the I2V model operates in fixed-resolution mode, generating outputs at the user-specified height and width. |
| `--max_hxw` 236544                             | When `crop_for_hw` is not specified, the I2V model operates in arbitrary resolution mode, resizing outputs to the greatest common divisor of the resolutions in the input image list. In this case, the `--max_hxw` parameter must be provided, with a default value of 236544. |
| `--text_prompt`                                | The path to the `prompt` file, where each line represents a prompt. Each line must correspond precisely to each line in `--conditional_pixel_values_path`. |
| `--conditional_pixel_values_path`              | The input path for control information can contain one or multiple images or videos, with each line controlling the generation of one video. It must correspond precisely to each prompt in `--text_prompt`. |
| `--mask_type`                                  | Specify the mask type used for the current inference; available types are listed in the `MaskType` class in `opensora/utils/mask_utils.py`, which are six mask types: `t2iv`, `i2v`, `transition`, `continuation`, `clear`, and `random_temporal`. This parameter can be omitted when performing I2V and Transition tasks. |

Before inference, you need to create two text files: one named `prompt.txt` and another named `conditional_pixel_values_path`. Each line of text in `prompt.txt` should correspond to the paths on each line in `conditional_pixel_values_path`.

For example, if the content of `prompt.txt` is:

```
this is a prompt of i2v task.
this is a prompt of transition task.
```

Then the content of `conditional_pixel_values_path` should be:

```
/path/to/image_0.png
/path/to/image_1_0.png,/path/to/image_1_1.png
```

This means we will execute a image-to-video task using `/path/to/image_0.png` and "this is a prompt of i2v task." For the transition task, we'll use `/path/to/image_1_0.png` and `/path/to/image_1_1.png` (note that these two paths are separated by a comma without any spaces) along with "this is a prompt of transition task."

After creating the files, make sure to specify their paths in the `sample_inpaint_v1_3.sh` script.