export WANDB_API_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
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
export TOKENIZERS_PARALLELISM=false
# export NCCL_ALGO=Tree


accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero1_config.yaml \
    opensora/train/train_t2v_diffusers.py \
    --ema_deepspeed_config_file scripts/accelerate_configs/zero3.json \
    --model OpenSoraT2V_v1_5-2B/122 \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data scripts/train_data/image_data_debug.txt \
    --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_height 256 \
    --max_width 256 \
    --force_resolution \
    --train_batch_size=32 \
    --train_image_batch_size=1 \
    --dataloader_num_workers 16 \
    --learning_rate=1e-4 \
    --adam_epsilon 1e-15 \
    --adam_weight_decay 1e-4 \
    --lr_scheduler="constant_with_warmup" \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=2000 \
    --gradient_checkpointing \
    --model_max_length 512 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --resume_from_checkpoint="latest" \
    --ema_update_freq 1 \
    --ema_decay 0.999 \
    --drop_short_ratio 0.0 \
    --hw_stride 16 \
    --train_fps 16 \
    --seed 1234 \
    --group_data \
    --use_decord \
    --output_dir="debug_train_log_mmdit_all" \
    --vae_fp32 \
    --rf_scheduler \
    --proj_name "debug_train_log" \
    --log_name mmdit_all \
    --trained_data_global_step 0 \
    --skip_abnorml_step --ema_decay_grad_clipping 0.99 