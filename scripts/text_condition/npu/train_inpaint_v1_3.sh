
export PROJECT=$PROJECT_NAME
# export PROJECT='test'
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

export TASK_QUEUE_ENABLE=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export HCCL_ALGO="level0:NA;level1:H-D_R"
# --machine_rank=${MACHINE_RANK} \
# --main_process_ip=${MAIN_PROCESS_IP_VALUE} \
# multi_node_example_by_deepspeed.yaml
# deepspeed_zero2_config.yaml

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint_v1_3-2B/122 \
    --text_encoder_name_1 google/mt5-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset inpaint \
    --data "scripts/train_data/current_hq_on_npu.txt" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/save_dir/lzj/formal_8dim/latent8" \
    --vae_fp32 \
    --sample_rate 1 \
    --num_frames 93 \
    --max_hxw 236544 \
    --min_hxw 102400 \
    --snr_gamma 5.0 \
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
    --checkpointing_steps=500 \
    --allow_tf32 \
    --model_max_length 512 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --speed_factor 1.0 \
    --ema_decay 0.9999 \
    --drop_short_ratio 0.0 \
    --hw_stride 32 \
    --sparse1d --sparse_n=4 \
    --train_fps 16 \
    --seed 1234 \
    --trained_data_global_step 0 \
    --group_data \
    --use_decord \
    --prediction_type "v_prediction" \
    --output_dir="/home/save_dir/runs/$PROJECT" \
    --mask_config scripts/train_configs/mask_config.yaml \
    --default_text_ratio 0.5 \
    --resume_from_checkpoint="latest" \
    # --pretrained_transformer_model_path /home/save_dir/pretrained/i2v_ckpt14777_ema \
    # --force_resolution
    # --force_resolution \
    # --max_height 352 \
    # --max_width 640 \
