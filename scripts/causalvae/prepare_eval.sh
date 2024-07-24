export CUDA_VISIBLE_DEVICES=0
REAL_DATASET_DIR=valid/
EXP_NAME=decoder
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=512
SUBSET_SIZE=1
CKPT=/storage/lcm/Causal-Video-VAE/results/488dim8

python opensora/models/causalvideovae/sample/rec_video_vae.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir valid_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --crop_size ${RESOLUTION} \
    --num_workers 8 \
    --ckpt ${CKPT} \
    --output_origin \
    --enable_tiling