WEIGHT_PATH="/home/opensora/pre_weights/"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export MASTER_PORT=12359

if [ -z "$SAMPLE_SAVE_PATH" ]; then
  export SAMPLE_SAVE_PATH="/home/image_data/sample_videos"
fi

if [ -z "$SAMPLE_HEIGHT" ]; then
  echo "You should set both envs of SAMPLE_HEIGHT and SAMPLE_WIDTH"
  return
fi

torchrun --nproc_per_node=8 opensora/sample/sample_inpaint_on_npu.py \
    --model_path /home/image_data/checkpoints/${PROJECT_NAME} \
    --num_frames ${NUM_FRAME} \
    --height $SAMPLE_HEIGHT \
    --width $SAMPLE_WIDTH \
    --cache_dir "../cache_dir" \
    --text_encoder_name ${WEIGHT_PATH}/google/mt5-xxl \
    --text_prompt "/home/image_data/checkpoints/i2v_validation_dir/prompt.txt" \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/test140k/" \
    --save_img_path "${SAMPLE_SAVE_PATH}/${PROJECT_NAME}" \
    --fps 24 \
    --max_sequence_length 512 \
    --guidance_scale 5.0 \
    --num_sampling_steps 50 \
    --sample_method PNDM \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --model_3d \
    --validation_dir "/home/image_data/checkpoints/i2v_validation_dir"