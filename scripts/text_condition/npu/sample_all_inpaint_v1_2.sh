
export TASK_QUEUE_ENABLE=0
torchrun --nnodes=1 --nproc_per_node 1 --master_port 29522 \
    -m opensora.sample.sample \
    --model_type "inpaint" \
    --model_path /home/save_dir/runs/all_inpaint_6/checkpoint-25339/model_ema \
    --version v1_2 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --max_hw_square 1048576 \
    --crop_for_hw \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/save_dir/pretrained/mt5-xxl" \
    --text_prompt /home/image_data/gyy/suv/all_inpaint_prompt.txt \
    --conditional_pixel_values_path /home/image_data/gyy/suv/all_inpaint_cond_imgs_path.txt \
    --mask_path /home/image_data/gyy/suv/all_inpaint_mask_path.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/save_dir/lzj/wf-vae_trilinear" \
    --save_img_path "./test_all_inpaint6" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --motion_score 0.95 \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr \
    --mask_type semantic \
    # --enable_tiling 
