export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
# export WANDB_MODE="offline"
export ENTITY="yunyangge"
export PROJECT="inpaint_93x480p_bs4x8x1_lr1e-5_snr5_noioff0.02_new_mask_onlysucai"
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
# export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_ALGO=Tree

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "/storage/ongoing/new/cache_dir" \
    --dataset i2v \
    --video_data "scripts/train_data/video_data_aesmovie_sucai.txt" \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --sample_rate 1 \
    --num_frames 93 \
    --use_image_num 0 \
    --max_height 720 \
    --max_width 1280 \
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
    --tile_overlap_factor 0.125 \
    --enable_tiling \
    --snr_gamma 5.0 \
    --use_ema  \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --use_rope \
    --resume_from_checkpoint="latest" \
    --ema_decay 0.999 \
    --group_frame \
    --speed_factor 1.1 \
    --pretrained "/storage/gyy/hw/Open-Sora-Plan/inpaint_93x480p_bs4x8x1_lr1e-5_snr5_noioff0.02_new_mask/checkpoint-3400/model_ema/diffusion_pytorch_model.safetensors" \
    --output_dir=$PROJECT \
    --i2v_ratio 0.5 \
    --transition_ratio 0.4 \
    --default_text_ratio 0.5 \
    --validation_dir "validation_dir" \
    --num_sampling_steps 50 \
    --guidance_scale 5.0 \
    --seed 42
