export PROJECT=$PROJECT_NAME
WEIGHT_PATH="/home/opensora/pre_weights/"
env
#export WANDB_MODE='offline'
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_ALGO="level0:NA;level1:H-D_R"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example_by_deepspeed.yaml \
    --machine_rank=${MACHINE_RANK} \
    --main_process_ip=${MAIN_PROCESS_IP_VALUE} \
    opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V-ROPE-L/122 \
    --text_encoder_name ${WEIGHT_PATH}/google/mt5-xxl \
    --cache_dir "../cache_dir" \
    --dataset t2v \
    --data "scripts/train_data/merge_data_on_npu.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "${WEIGHT_PATH}/test140k/" \
    --sample_rate 1 \
    --num_frames ${NUM_FRAME} \
    --max_height 720 \
    --max_width 1280 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.5 \
    --interpolation_scale_w 2.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --seed=10 \
    --lr_warmup_steps=500 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1000 \
    --output_dir="/home/image_data/checkpoints/${PROJECT}/" \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --use_rope \
    --noise_offset 0.02 \
    --resume_from_checkpoint="latest" \
    --enable_stable_fp32 \
    --ema_decay 0.999 \
    --speed_factor 1.0 \
    --drop_short_ratio 1.0
