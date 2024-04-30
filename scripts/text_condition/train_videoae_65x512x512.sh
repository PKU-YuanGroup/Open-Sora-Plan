export WANDB_MODE='offline'
export ENTITY=""
export PROJECT="512"
WEIGHT_PATH="/home/opensora/shebin/pre_weights/"
export http_proxy="http://192.168.0.30:8888"
export https_proxy="http://192.168.0.30:8888"
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name ${WEIGHT_PATH}/DeepFloyd/t5-v1_1-xxl \
    --cache_dir "../cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/CausalVAEModel_4x8x8_new_version/" \
    --video_data "./scripts/train_data/video_data.txt" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=2 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=100 \
    --output_dir="512" \
    --allow_tf32 \
    --pretrained "${WEIGHT_PATH}/t2v.pt" \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --enable_tiling \
    --use_img_from_vid \
    --enable_tracker
