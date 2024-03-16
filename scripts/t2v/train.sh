
accelerate launch \
    --config_file scripts/accelerate_configs/ddp_config.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --dataset t2v \
    --ae stabilityai/sd-vae-ft-mse \
    --data_path /root/autodl-tmp/sea.csv \
    --video_folder /root/autodl-tmp/sea \
    --extras 3 \
    --sample_rate 3 \
    --num_frames 16 \
    --max_image_size 256 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=5 --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="tensorboard" \
    --checkpointing_steps=500 \
    --output_dir="t2v-f16s3-256-imgvae188-bf16-ckpt-xformers"
