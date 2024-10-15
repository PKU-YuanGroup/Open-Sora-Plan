
### Data prepare
The organization of the training data is easy. We only need to put all the videos recursively in a directory. This makes the training more convenient when using multiple datasets.
``` shell
Training Dataset
|——sub_dataset1
    |——sub_sub_dataset1
        |——video1.mp4
        |——video2.mp4
        ......
    |——sub_sub_dataset2
        |——video3.mp4
        |——video4.mp4
        ......
|——sub_dataset2
    |——video5.mp4
    |——video6.mp4
    ......
|——video7.mp4
|——video8.mp4
```

### Training
``` shell
bash scripts/causalvae/train.sh
```
We introduce the important args for training.

| Argparse | Usage |
|:---|:---|
|_Training size_||
|`--num_frames`|The number of using frames for training videos|
|`--resolution`|The resolution of the input to the VAE|
|`--batch_size`|The local batch size in each GPU|
|`--sample_rate`|The frame interval of when loading training videos|
|_Data processing_||
|`--video_path`|/path/to/dataset|
|_Load weights_||
|`--model_name`| `CausalVAE` or `WFVAE`|
|`--model_config`|/path/to/config.json The model config of VAE. If you want to train from scratch use this parameter.|
|`--pretrained_model_name_or_path`|A directory containing a model checkpoint and its config. Using this parameter will only load its weight but not load the state of the optimizer|
|`--resume_from_checkpoint`|/path/to/checkpoint It will resume the training process from the checkpoint including the weight and the optimizer.|

### Inference

``` shell
bash scripts/causalvae/rec_video.sh
```
We introduce the important args for inference.
| Argparse | Usage |
|:---|:---|
|_Ouoput video size_||
|`--num_frames`|The number of frames of generated videos|
|`--height`|The resolution of generated videos|
|`--width`|The resolution of generated videos|
|_Data processing_||
|`--video_path`|The path to the original video|
|`--rec_path`|The path to the generated video|
|_Load weights_||
|`--ae_path`|/path/to/model_dir. A directory containing the checkpoint of VAE is used for inference and its model config.json|
|_Other_||
|`--enable_tilintg`|Use tiling to deal with videos of high resolution and long duration|
|`--save_memory`|Save memory to inference but lightly influence quality|


### Evaluation

The evaluation process consists of two steps:

Reconstruct videos in batches: `bash scripts/causalvae/prepare_eval.sh`
Evaluate video metrics: `bash scripts/causalvae/eval.sh`

To simplify the evaluation, environment variables are used for control. For step 1 (`bash scripts/causalvae/prepare_eval.sh`):

```bash
# Experiment name
EXP_NAME=wfvae
# Video parameters
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
# Model weights
CKPT=ckpt
# Select subset size (0 for full set)
SUBSET_SIZE=0
# Dataset directory
DATASET_DIR=test_video
```

For step 2 (`scripts/causalvae/eval.sh`):

```bash
# Experiment name
EXP_NAME=wfvae-4dim
# Video parameters
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
# Evaluation metric
METRIC=lpips
# Select subset size (0 for full set)
SUBSET_SIZE=0
# Path to the ground truth videos, which can be saved during video reconstruction by setting `--output_origin`
ORIGIN_DIR=video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE}/origin
# Path to the reconstructed videos
RECON_DIR=video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE}
```