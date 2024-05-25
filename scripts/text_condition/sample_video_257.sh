CUDA_VISIBLE_DEVICES=2 python opensora/sample/sample_t2v.py \
    --model_path 257f_512_10node_bs1_lr2e-5_8img/checkpoint-5000/model \
    --version 65x512x512 \
    --image_size 512 \
    --cache_dir "./cache_dir" \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/dxyl_data02/CausalVAEModel_4x8x8/" \
    --save_img_path "./sample_videos_257f_5k_5.5_100s" \
    --fps 24 \
    --guidance_scale 5.5 \
    --num_sampling_steps 100 \
    --enable_tiling