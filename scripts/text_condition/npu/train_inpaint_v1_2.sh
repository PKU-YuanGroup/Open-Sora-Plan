export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
export WANDB_MODE="online"
export ENTITY="yunyang"
export PROJECT=$PROJECT_NAME
# export PROJECT='test'
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

export TASK_QUEUE_ENABLE=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
# export HCCL_ALGO="level0:NA;level1:H-D_R"
# --machine_rank=${MACHINE_RANK} \ 
# --main_process_ip=${MAIN_PROCESS_IP_VALUE} \ 

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint_v1_2-L/122 \
    --text_encoder_name_1 google/mt5-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset inpaint \
    --data "scripts/train_data/debug_on_npu.txt" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/image_data/lb/Open-Sora-Plan/WFVAE_DISTILL_FORMAL" \
    --vae_fp32 \
    --sample_rate 1 \
    --num_frames 93 \
    --max_height 512 \
    --max_width 512 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
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
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --skip_low_resolution \
    --speed_factor 1.0 \
    --ema_decay 0.9999 \
    --drop_short_ratio 0.0 \
    --hw_stride 32 \
    --sparse1d --sparse_n 4 \
    --use_motion \
    --train_fps 16 \
    --seed 1234 \
    --trained_data_global_step 0 \
    --group_data \
    --use_decord \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr \
    --output_dir="debug" \
    --t2v_ratio 0.02 \
    --i2v_ratio 0.5 \
    --transition_ratio 0.3 \
    --v2v_ratio 0.0 \
    --clear_video_ratio 0.0 \
    --min_clear_ratio 0.0 \
    --max_clear_ratio 1.0 \
    --default_text_ratio 0.5 \
    --pretrained_transformer_model_path "/home/image_data/captions/vpre_latest_168k/model_ema" \
    # --resume_from_checkpoint="latest" \
    # --min_height 0 \
    # --min_width 0 \
