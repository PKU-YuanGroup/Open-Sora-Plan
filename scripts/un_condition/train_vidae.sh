export WANDB_KEY=""
export ENTITY=""
export PROJECT="sky-f17s3-128-causalvideovae444-bf16-ckpt-flash-log"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch \
    --config_file scripts/accelerate_configs/ddp_config.yaml \
    opensora/train/train.py \
    --model Latte-XL/122 \
    --dataset sky \
    --ae CausalVQVAEModel_4x4x4 \
    --data_path /remote-home/yeyang/sky_timelapse/sky_train/ \
    --sample_rate 3 \
    --num_frames 17 \
    --max_image_size 128 \
    --gradient_checkpointing \
    --attention_mode flash \
    --train_batch_size=8 --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="sky-f17s3-128-causalvideovae444-bf16-ckpt-flash-log" \
    --allow_tf32

