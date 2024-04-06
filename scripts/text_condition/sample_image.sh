export ASCEND_RT_VISIBLE_DEVICES=0
python opensora/sample/sample_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt "A quiet beach at dawn, the waves gently lapping at the shore and the sky painted in pastel hues." \
    --ae CausalVAEModel_4x8x8 \
    --ckpt checkpoint-6500 \
    --fps 24 \
    --num_frames 65 \
    --image_size 512 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --force_images
