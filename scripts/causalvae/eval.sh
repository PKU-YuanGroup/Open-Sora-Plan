# REAL_DATASET_DIR=/remote-home1/dataset/OpenMMLab___Kinetics-400/raw/Kinetics-400/videos_val/
REAL_DATASET_DIR=../dataset/webvid/videos
EXP_NAME=decoder
SAMPLE_RATE=3
NUM_FRAMES=33
RESOLUTION=256
SUBSET_SIZE=50
METRIC=ssim

python opensora/models/causalvideovae/eval/eval_common_metric.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir /remote-home1/lzj/dataset/gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda:0 \
    --sample_fps 3 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --crop_size ${RESOLUTION} \
    --metric ${METRIC}