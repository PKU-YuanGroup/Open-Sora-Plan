export WANDB_KEY="c54943d667ed1abb58ed994e739462e66bc1aee2"
export WANDB_MODE="online"
export ENTITY="hexianyi"
export PROJECT=$PROJECT_NAME
# export PROJECT='test'
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
# export HCCL_ALGO="level0:NA;level1:H-D_R"

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example_by_deepspeed.yaml \
    --machine_rank=${MACHINE_RANK} \
    --main_process_ip=${MAIN_PROCESS_IP_VALUE} \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset inpaint \
    --data "scripts/train_data/video_data_debug.txt" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/image_data/lb/Open-Sora-Plan/WFVAE_DISTILL_FORMAL" \
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
    --dataloader_num_workers 0 \
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
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.0 \
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
    --trained_data_global_step 0 \
    --group_data \
    --use_decord \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr \
    --t2v_ratio 0.0 \
    --i2v_ratio 0.0 \
    --transition_ratio 0.0 \
    --v2v_ratio 0.0 \
    --Semantic_ratio 0.2\
    --bbox_ratio 0.2\
    --background_ratio 0.2\
    --fixed_ratio 0.1\
    --Semantic_expansion_ratio 0.1\
    --fixed_bg_ratio 0.1\
    --clear_video_ratio 0.0 \
    --min_clear_ratio 0.25 \
    --default_text_ratio 0.0 \
    --output_dir /home/save_dir/runs/$PROJECT \
    --pretrained_transformer_model_path "/home/image_data/captions/vpre_latest_134k/model_ema" \
    --yolomodel_pathorname "/home/image_data/hxy/Open-Sora-Plan/opensora/dataset/yolov9c-seg.pt"\
    # --resume_from_checkpoint="/home/save_dir/runs/allinpaint_stage1/checkpoint-13000" 
    # 切part是resume，不是pretrained
    # --snr_gamma 5.0 \
