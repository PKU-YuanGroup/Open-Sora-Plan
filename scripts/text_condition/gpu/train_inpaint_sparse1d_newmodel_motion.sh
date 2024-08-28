export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="online"
export ENTITY="yunyang"
export PROJECT="test_sparse_inpaint"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=25
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_RETRY_CNT=32
# export NCCL_ALGO=Tree

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset inpaint \
    --data "scripts/train_data/merge_data_debug.txt" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/Causal-Video-VAE/results/WFVAE_DISTILL_FORMAL" \
    --sample_rate 1 \
    --num_frames 93 \
    --max_height 320 \
    --max_width 320 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1000 \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --use_rope \
    --skip_low_resolution \
    --speed_factor 1.0 \
    --ema_decay 0.9999 \
    --drop_short_ratio 0.0 \
    --hw_stride 32 \
    --sparse1d --sparse_n 4 \
    --use_motion \
    --train_fps 16 \
    --seed 1234 \
    --group_data \
    --t2v_ratio 0.1 \
    --i2v_ratio 0.0 \
    --transition_ratio 0.0 \
    --v2v_ratio 0.0 \
    --clear_video_ratio 0.0 \
    --min_clear_ratio 0.5 \
    --default_text_ratio 0.5 \
    --pretrained_transformer_model_path "/storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs32x8x1_anyx93x320x320_fps16_lr1e-5_snr5_noioff0.02_ema9999_sparse1d4_dit_l_mt5xxl_alldata100m/checkpoint-526000/model_ema" \
    --output_dir="test_sparse_inpaint"  > training_log_new.txt
    # --resume_from_checkpoint="latest" \
