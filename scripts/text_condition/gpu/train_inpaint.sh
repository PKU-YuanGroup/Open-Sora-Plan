PROJECT="your project name"
export WANDB_API_KEY="your wanb key"
export WANDB_MODE="online"
export ENTITY="your name"
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

# export NCCL_ALGO=Tree

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "../cache_dir" \
    --dataset inpaint \
    --data "scripts/train_data/merge_data.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --sample_rate 1 \
    --num_frames 93 \
    --use_image_num 0 \
    --max_height 480 \
    --max_width 640 \
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
    --checkpointing_steps=200 \
    --allow_tf32 \
    --model_max_length 512 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --i2v_ratio 0.4 \
    --transition_ratio 0.4 \
    --v2v_ratio 0.1 \
    --clear_video_ratio 0.0 \
    --default_text_ratio 0.5 \
    --use_rope \
    --ema_decay 0.999 \
    --speed_factor 1.0 \
    --group_frame \
    --output_dir runs/$PROJECT \
    --pretrained_transformer_model_path "/path/to/pretrained_model" \
    # --resume_from_checkpoint="latest" \
