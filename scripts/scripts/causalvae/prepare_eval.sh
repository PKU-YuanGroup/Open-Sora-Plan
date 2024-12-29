export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATASET_DIR=test_video
EXP_NAME=wfvae
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
CKPT=ckpt
SUBSET_SIZE=0

accelerate launch \
    --config_file scripts/accelerate_configs/default_config.yaml \
    opensora/models/causalvideovae/sample/rec_video_vae.py \
    --batch_size 1 \
    --real_video_dir ${DATASET_DIR} \
    --generated_video_dir video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --num_workers 8 \
    --from_pretrained ${CKPT} \
    --model_name WFVAE \
    --output_origin \
    --crop_size ${RESOLUTION}
