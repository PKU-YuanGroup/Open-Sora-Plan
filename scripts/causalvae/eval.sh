EXP_NAME=wfvae-4dim
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
METRIC=lpips
SUBSET_SIZE=0
ORIGIN_DIR=video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE}/origin
RECON_DIR=video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE}

python opensora/models/causalvideovae/eval/eval.py \
    --batch_size 8 \
    --real_video_dir ${ORIGIN_DIR} \
    --generated_video_dir ${RECON_DIR} \
    --device cuda:1 \
    --sample_fps 1 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --crop_size ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --metric ${METRIC}