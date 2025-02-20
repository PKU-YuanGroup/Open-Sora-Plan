
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.sample.sample \
    --model_type "inpaint" \
    --model_path model_path \
    --version v1_3 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --max_hxw 236544 \
    --crop_for_hw \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl" \
    --text_prompt examples/cond_prompt.txt \
    --conditional_pixel_values_path examples/cond_pix_path.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/latent8" \
    --save_img_path "./save_path" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --rescale_betas_zero_snr \
    --prediction_type "v_prediction" \
    --noise_strength 0.0 \