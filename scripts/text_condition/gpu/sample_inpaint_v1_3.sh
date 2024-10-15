
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.sample.sample \
    --model_type "inpaint" \
    --model_path /storage/gyy/hw/Open-Sora-Plan/runs/inpaint_93x1280x1280_stage3_gpu/checkpoint-1692/model_ema \
    --version v1_3 \
    --num_frames 93 \
    --height 704 \
    --width 1280 \
    --max_hw_square 1048576 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl" \
    --text_prompt /storage/gyy/hw/Open-Sora-Plan/test_prompt.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/wf-vae_trilinear" \
    --save_img_path "./test_inpaint" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr \
    --conditional_images_path /storage/gyy/hw/Open-Sora-Plan/test_cond_imgs_path.txt \
    --sp \
    --enable_tiling 


export TASK_QUEUE_ENABLE=0
torchrun --nnodes=1 --nproc_per_node 8 --master_port 29522 \
    -m opensora.sample.sample \
    --model_type "inpaint" \
    --model_path "" \
    --version v1_3 \
    --num_frames 93 \
    --max_hxw 236544 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl" \
    --text_prompt examples/cond_prompt.txt \
    --conditional_pixel_values_path examples/cond_prompt.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/latent8" \
    --save_img_path "./test_inpaint_v1_3" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    # --crop_for_hw \
    # --height 352 \
    # --width 640 \
    # --mask_type i2v \
    # --enable_tiling 
