export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
export ENTITY="linbin"
export PROJECT="bs32x8x2_61x480p_lr1e-4_snr5_noioff0.02_opensora122_rope_mt5xxl_pandamovie_aes_mo_sucai_mo_speed1.2"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
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
# export NCCL_ALGO=Tree

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data "scripts/train_data/merge_data_mj.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/storage/dataset/488dim4_plus" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_height 720 \
    --max_width 1280 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.5 \
    --interpolation_scale_w 2.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=16 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1000 \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --tile_overlap_factor 0.125 \
    --enable_tiling \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --use_rope \
    --resume_from_checkpoint="latest" \
    --group_data \
    --skip_low_resolution \
    --speed_factor 1.0 \
    --ema_decay 0.9999 \
    --drop_short_ratio 0.0 \
    --sparse1d \
    --sparse_n 16 \
    --force_resolution \
    --enable_tracker \
    --pretrained "/storage/dataset/hw29/image/model_ema/diffusion_pytorch_model.safetensors" \
    --output_dir="bs1x8x16_1x720p_lr1e-4_snr5_noioff0.02_ema9999_sparse1d16_dit_l_122_rope_mt5xxl_mj_fromhw480p" 
