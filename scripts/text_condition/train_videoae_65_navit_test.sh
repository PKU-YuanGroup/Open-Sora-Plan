export WANDB_KEY=""
export ENTITY="linbin"
export PROJECT="65x512x512_10node_bs2_lr2e-5_4img"
export CUDA_VISIBLE_DEVICES=1
python tests/test_navit_consistency.py \
    --model NaViTLatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/mnt_zhexin/zhexin.lzx/models/Open-Sora-Plan-v1.1.0/vae/" \
    --video_data "scripts/train_data/video_data_debug.txt" \
    --image_data "scripts/train_data/image_data_debug.txt" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=4 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="65x512x512_10node_bs2_lr2e-5_4img" \
    --allow_tf32 \
    --use_deepspeed \
    --max_token_lim 2048 \
    --token_dropout_rate 0.0 \
    --use_image_num 4 \
    --enable_tiling \
    --enable_tracker \
    --resume_from_checkpoint "latest" 
