WEIGHT_PATH="/home/opensora/shebin/pre_weights/"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

CUDA_VISIBLE_DEVICES=6 python opensora/sample/sample_t2v.py \
    --model_path 512/checkpoint-3500/model \
    --version 65x512x512 \
    --image_size 512 \
    --cache_dir "./cache_dir" \
    --text_encoder_name ${WEIGHT_PATH}/DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/CausalVAEModel_4x8x8_0430/" \
    --save_img_path "./sample_videos" \
    --fps 24 \
    --guidance_scale 10.0 \
    --num_sampling_steps 50 \
    --enable_tiling
