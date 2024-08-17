# PROJECT="video_test"
PROJECT="vip_inpaint_480p_f93_bs4x8x1_lr1e-5_snrgamma5_0_noiseoffset0_02_ema0_999_all_from_scratch"
export WANDB_API_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="online"
export ENTITY="yunyangge"
export PROJECT=$PROJECT
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PDSH_RCMD_TYPE=ssh

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example1.yaml \
    opensora/train/train_inpaint_all_in_one.py \
    --model OpenSoraInpaint-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --image_encoder_name vit_giant_patch14_reg4_dinov2.lvd142m \
    --image_encoder_path /storage/cache_dir/hub/models--timm--vit_giant_patch14_reg4_dinov2.lvd142m/snapshots/a2208b21b069f6b2e45999870fcce4b7e43d1a2c/model.safetensors \
    --cache_dir "/storage/cache_dir" \
    --dataset inpaint \
    --model_type vip_inpaint \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --data "scripts/train_data/video_data.txt" \
    --sample_rate 1 \
    --num_frames 93 \
    --use_image_num 0 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --train_batch_size=1 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_train_steps=1000000 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --enable_tracker \
    --checkpointing_steps=200 \
    --output_dir runs/$PROJECT \
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
    --v2v_ratio 0.1 \
    --clear_video_ratio 0.0 \
    --default_text_ratio 0.5 \
    --seed 42 \
    --snr_gamma 5.0 \
    --noise_offset 0.02 \
    --vip_num_attention_heads 16 \
    --ema_decay 0.999 \
    --use_rope \
    --group_frame \
    --train_vip \
    --pretrained_transformer_model_path "/storage/ongoing/new/Open-Sora-Plan-bak/7.14bak/bs16x8x1_93x480p_lr1e-4_snr5_ema999_opensora122_rope_mt5xxl_high_pandamovie_speed1.0/checkpoint-3500/model_ema" \
    # --pretrained_vip_adapter_path "/storage/gyy/hw/Open-Sora-Plan/pretrained_models/pretrained_vip_9000.pth" \
    # --resume_from_checkpoint "latest" \
    # --speed_factor 1.5 \
    # --zero_terminal_snr \
    # 基模型权重没有参与训练所以一定要加载