export WANDB_KEY=""
export ENTITY=""
export PROJECT="t2v-f17s3-img4-128-causalvideovae444-bf16-ckpt-xformers"
accelerate launch \
    --config_file scripts/accelerate_configs/ddp_config.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --dataset t2v \
    --ae CausalVQVAEModel_4x4x4 \
    --data_path /root/autodl-tmp/sea.csv \
    --video_folder /root/autodl-tmp/sea \
    --sample_rate 3 \
    --num_frames 17 \
    --max_image_size 128 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=8 --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="t2v-f17s3-img4-128-causalvideovae444-bf16-ckpt-xformers" \
    --allow_tf32 \
    --use_image_num 4 \
    --use_img_from_vid \
    --pretrained t2v.pt \
    --use_deepspeed \
    --model_max_length 300


