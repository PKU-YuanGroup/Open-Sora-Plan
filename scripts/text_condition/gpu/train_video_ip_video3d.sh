PROJECT="videoip_3d_480p_bs8x16_lr1e-5_snrgamma5_0_noiseoffset0_02_dino518_ema0_999"
export WANDB_API_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="offline"
export ENTITY="yunyangge"
export PROJECT=$PROJECT
# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# # NCCL setting IB网卡时用
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1

export PDSH_RCMD_TYPE=ssh

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_videoip.py \
    --model OpenSoraT2V-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --image_encoder_name vit_giant_patch14_reg4_dinov2.lvd142m \
    --cache_dir "/storage/cache_dir" \
    --dataset vip \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/CausalVAEModel_4x8x8" \
    --video_data "scripts/train_data/video_data.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --sample_rate 1 \
    --num_frames 1 \
    --use_image_num 0 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --train_batch_size=16 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=500000 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --enable_tracker \
    --checkpointing_steps=1000 \
    --output_dir=$PROJECT \
    --allow_tf32 \
    --model_max_length 512 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --validation_dir "validation_dir" \
    --guidance_scale 5.0 \
    --num_sampling_steps 50 \
    --ema_start_step 0 \
    --use_ema \
    --cfg 0.05 \
    --i2v_ratio 0.4 \
    --transition_ratio 0.4 \
    --clear_video_ratio 0.1 \
    --default_text_ratio 0.5 \
    --seed 42 \
    --snr_gamma 5.0 \
    --noise_offset 0.02 \
    --vip_num_attention_heads 16 \
    --ema_decay 0.999 \
    --use_rope \
    --pretrained "/storage/dataset/hw29/image/model_ema/diffusion_pytorch_model.safetensors" \
    --resume_from_checkpoint "latest" \
    # --pretrained_vip_adapter_path "videoip_65x512x512_1node_bs32_lr1e-5_snrgamma_noiseoffset_mjsam_clip_last2layer/checkpoint-50000/model" \
    # --zero_terminal_snr \
    # 基模型权重没有参与训练所以一定要加载