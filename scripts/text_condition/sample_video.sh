export MUSA_LAUNCH_BLOCKING=0

CUDA_VISIBLE_DEVICES=0 python opensora/sample/sample_t2v.py \
    --model_path /home/dist/haoran/Open-Sora-Plan-v1.0.0 \
    --text_encoder_name /home/dist/haoran/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --version 65x256x256 \
    --save_img_path "./sample_videos/prompt_list_0" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 250 \
    --enable_tiling
