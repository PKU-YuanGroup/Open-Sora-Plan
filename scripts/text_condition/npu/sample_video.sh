WEIGHT_PATH="/home/opensora/pre_weights/"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export MASTER_PORT=12359

if [ -z "$SAMPLE_SAVE_PATH" ]; then
  export SAMPLE_SAVE_PATH="/home/image_data/sample_videos"
fi

torchrun --nproc_per_node=8 opensora/sample/sample_t2v_on_npu.py \
    --model_path /home/image_data/checkpoints/${PROJECT_NAME} \
    --num_frames ${NUM_FRAME} \
    --height 480 \
    --width 640 \
    --cache_dir "./cache_dir" \
    --text_encoder_name ${WEIGHT_PATH}/DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/test140k/" \
    --save_img_path "${SAMPLE_SAVE_PATH}/${PROJECT_NAME}" \
    --fps 24 \
    --guidance_scale 2.0 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --sample_method DDPM \
    --model_3d