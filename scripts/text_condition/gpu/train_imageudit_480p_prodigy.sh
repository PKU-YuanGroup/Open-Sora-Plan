export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
export ENTITY="linbin"
export PROJECT="bs32_2node_lr1e-4_snr5_ema_ps11_ds11_udit22_prodigy"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
# NCCL setting
export PDSH_RCMD_TYPE=ssh
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example1.yaml \
    opensora/train/train_t2v_diffusers.py \
    --model UDiTT2V-L/122 \
    --text_encoder_name google/umt5-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/CausalVAEModel_4x8x8/" \
    --video_data "scripts/train_data/video_data.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=32 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --optimizer="prodigy" \
    --learning_rate=1.0 \
    --prodigy_safeguard_warmup=True \
    --prodigy_use_bias_correction=True \
    --adam_beta1=0.9 \
    --adam_beta2=0.99  \
    --adam_weight_decay=0.01 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="bs32_2node_480p_lr1e-4_snr5_ema_ps11_ds11_udit22_prodigy_umt5" \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --enable_tracker \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --downsampler "k33_s22"  \
    --resume_from_checkpoint="latest"
