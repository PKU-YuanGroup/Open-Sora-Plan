export WANDB_KEY=""
export ENTITY=""
export PROJECT="1024"
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-D64-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "../../Open-Sora-Plan/cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "../../Open-Sora-Plan/CausalVAEModel_4x8x8/" \
    --video_data_path "../../Open-Sora-Plan/sharegpt4v_path_cap_64x512x512_mixkit.json" \
    --video_folder /remote-home1/dataset/data_split_tt \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 1024 \
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
    --report_to="tensorboard" \
    --checkpointing_steps=100 \
    --output_dir="1024" \
    --allow_tf32 \
    --pretrained t2v.pt \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --use_img_from_vid \
    --enable_tiling 
