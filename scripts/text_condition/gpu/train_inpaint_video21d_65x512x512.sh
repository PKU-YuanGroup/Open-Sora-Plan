# export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# # export WANDB_MODE="offline"
# export ENTITY="linbin"
# export PROJECT="testnpu21d_"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
# # NCCL setting IB网卡时用
# export NCCL_PXN_DISABLE=0
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_GID_INDEX=3
# export NCCL_ALGO=Ring
# export OMP_NUM_THREADS=1

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint_XL_122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "/data01/transition/cache_dir" \
    --dataset inpaint \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/data01/transition/Open-Sora-Plan_models/vae" \
    --video_data "scripts/train_data/video_data_debug_inpaint.txt" \
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
    --checkpointing_steps=200 \
    --output_dir="inpaint_65x512x512_1node_bs4_lr2e-5" \
    --allow_tf32 \
    --model_max_length 300 \
    --enable_tiling \
    --pretrained "/data01/transition/Open-Sora-Plan_models/65x512x512/diffusion_pytorch_model.safetensors" \
    --validation_dir "/data01/transition/hw/Open-Sora-Plan/validation_dir" \
    --use_image_num 0 \
    --cfg 0.1 \
    --ema_start_step 0 \
    --use_ema \
    --i2v_ratio 0.4 \
    --transition_ratio 0.4 \
    --default_text_ratio 0.1 \
    # --snr_gamma 5.0 \
    # --zero_terminal_snr \
    # --noise_offset 0.02 \