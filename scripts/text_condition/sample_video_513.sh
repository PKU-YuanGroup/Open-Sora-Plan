
CUDA_VISIBLE_DEVICES=4 python opensora/sample/sample_t2v.py \
    --model_path 513f_512_9node_bs1_lr2e-5_4img/checkpoint-11500/model \
    --version 65x512x512 \
    --num_frames 513 \
    --height 512 \
    --width 512 \
    --cache_dir "./cache_dir" \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/dxyl_data02/CausalVAEModel_4x8x8/" \
    --save_img_path "./testttt" \
    --fps 24 \
    --guidance_scale 5.0 \
    --num_sampling_steps 150 \
    --enable_tiling
