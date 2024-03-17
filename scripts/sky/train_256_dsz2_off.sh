
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_offload_config.yaml \
    opensora/train/train.py \
    --model Latte-XL/122 \
    --dataset sky \
    --ae stabilityai/sd-vae-ft-mse \
    --data_path /remote-home/yeyang/sky_timelapse/sky_train/ \
    --extras 1 \
    --sample_rate 1 \
    --num_frames 128 \
    --max_image_size 256 \
    --gradient_checkpointing \
    --attention_mode flash \
    --train_batch_size=5 --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="tensorboard" \
    --checkpointing_steps=500 \
    --output_dir="sky-f128s1-256-imgvae188-bf16-ckpt-flash-dsz2off" \
    --use_deepspeed
