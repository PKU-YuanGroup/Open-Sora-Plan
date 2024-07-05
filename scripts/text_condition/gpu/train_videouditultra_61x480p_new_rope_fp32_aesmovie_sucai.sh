export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
export ENTITY="linbin"
export PROJECT="bs2_4node_72500k_480p_61x480p_lr1e-4_snr5_noioff0.02_ema_rope_uditultra122_qknorm_ds122_mt5xxl_sucai288w"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
# NCCL setting
export PDSH_RCMD_TYPE=ssh
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Tree
export OMP_NUM_THREADS=1

accelerate launch \
    --config_file scripts/accelerate_configs/debug.yaml \
    opensora/train/train_t2v_diffusers.py \
    --model UDiTUltraT2V-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "../cache_dir" \
    --dataset t2v \
    --video_data "scripts/train_data/video_data_aesmovie_sucai.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --sample_rate 1 \
    --num_frames 61 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 2.0 \
    --interpolation_scale_w 2.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=2 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
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
    --downsampler "k333_s222"  \
    --resume_from_checkpoint="latest" \
    --enable_stable_fp32 \
    --group_frame \
    --speed_factor 1.2 \
    --pretrained "/storage/ongoing/new/Open-Sora-Plan/bs20x8x2_73k_480p_61x480p_lr1e-4_snr5_noioff0.02_rope_uditultra122_qknorm_ds222_mt5xxl_movie_aes_mo_sucai_mo_speed1.2/checkpoint-12500/model_ema/diffusion_pytorch_model.safetensors" \
    --output_dir="debug" 