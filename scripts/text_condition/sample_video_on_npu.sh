WEIGHT_PATH="/home/opensora/shebin/pre_weights/"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export MASTER_PORT=12359

torchrun --nproc_per_node=8 opensora/sample/sample_t2v.py \
    --model_path /home/image_data/checkpoints/${PROJECT_NAME} \
    --version ${NUM_FRAME}x512x512 \
    --num_frames 1 \
    --height 512 \
    --width 512 \
    --cache_dir "./cache_dir" \
    --text_encoder_name ${WEIGHT_PATH}/DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/CausalVAEModel_4x8x8_0430/" \
    --save_img_path "/home/image_data/yancen/sample_videos/${PROJECT_NAME}" \
    --fps 24 \
    --guidance_scale 4.0 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --model_3d
