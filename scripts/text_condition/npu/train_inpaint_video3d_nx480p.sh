export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
# # export WANDB_MODE="offline"
export ENTITY="yunyangge"
WEIGHT_PATH="/home/opensora/pre_weights/"
env
# export WANDB_MODE='offline'
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_ALGO="level0:NA;level1:H-D_R"

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example_by_deepspeed.yaml \
    --machine_rank=${MACHINE_RANK} \
    --main_process_ip=${MAIN_PROCESS_IP_VALUE} \
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint-ROPE-L/122 \
    --text_encoder_name ${WEIGHT_PATH}/google/mt5-xxl \
    --cache_dir "../cache_dir" \
    --dataset i2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/test140k/" \
    --video_data "./scripts/train_data/video_data_on_npu.txt" \
    --image_data "./scripts/train_data/image_data_on_npu.txt" \
    --sample_rate 1 \
    --num_frames ${NUM_FRAME} \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --seed=42 \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=200 \
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
    --pretrained "/home/image_data/checkpoints/gyy_pretrained_f125/model_ema/diffusion_pytorch_model.safetensors" \
    --resume_from_checkpoint="latest" \
    --i2v_ratio 0.5 \
    --transition_ratio 0.4 \
    --default_text_ratio 0.5 \
    --group_frame \
    --speed_factor 1.1 \
