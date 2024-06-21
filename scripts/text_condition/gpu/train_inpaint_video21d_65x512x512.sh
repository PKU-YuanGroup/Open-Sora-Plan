export WANDB_API_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
# # export WANDB_MODE="offline"
export ENTITY="yunyangge"
export PROJECT="inpaint_65x512x512_1node_bs2_lr2e-5"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# # NCCL setting IB网卡时用
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint_XL_122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "/storage/cache_dir" \
    --dataset inpaint \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/CausalVAEModel_4x8x8" \
    --video_data "scripts/train_data/video_data_debug.txt" \
    --sample_rate 1 \
    --num_frames 65\
    --max_height 512 \
    --max_width 512 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=2 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --enable_tracker \
    --checkpointing_steps=500 \
    --output_dir="inpaint_65x512x512_1node_bs2_lr2e-5" \
    --allow_tf32 \
    --model_max_length 300 \
    --enable_tiling \
    --pretrained "/storage/1.1model/hw_65/model/diffusion_pytorch_model.safetensors" \
    --validation_dir "validation_dir" \
    --use_image_num 0 \
    --cfg 0.1 \
    --ema_start_step 0 \
    --use_ema \
    --i2v_ratio 0.4 \
    --transition_ratio 0.4 \
    --default_text_ratio 0.1 \
    --seed 42 \
    # --snr_gamma 5.0 \
    # --zero_terminal_snr \
    # --noise_offset 0.02 \