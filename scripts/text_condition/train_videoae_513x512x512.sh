export WANDB_KEY=""
export ENTITY="linbin"
export PROJECT="513x512x512_10node_bs1_lr2e-5_4img"
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/remote-home1/yeyang/CausalVAEModel_4x8x8" \
    --video_data "scripts/train_data/video_data_513.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --sample_rate 1 \
    --num_frames 513 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=250 \
    --output_dir="513x512x512_10node_bs1_lr2e-5_4img" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --enable_tiling \
    --pretrained 221x512x512_10node_bs1_lr2e-5_4img/diffusion_pytorch_model.safetensors \
    --enable_tracker \
    --resume_from_checkpoint "latest"
