
export TASK_QUEUE_ENABLE=0
torchrun --nnodes=1 --nproc_per_node 8 --master_port 29522 \
    -m opensora.sample.sample \
    --model_type "inpaint" \
    --model_path /home/save_dir/runs/i2v_1_3_hq_finetune_from_org_sucai_movie_final/checkpoint-6600/model \
    --version v1_3 \
    --num_frames 93 \
    --crop_for_hw \
    --height 352 \
    --width 640 \
    --max_hxw 236544 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/save_dir/pretrained/mt5-xxl" \
    --text_prompt /home/image_data/gyy/mmdit/Open-Sora-Plan/validation_dir/prompt.txt \
    --conditional_pixel_values_path /home/image_data/gyy/mmdit/Open-Sora-Plan/validation_dir/cond_imgs_path.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/save_dir/lzj/formal_8dim/latent8" \
    --save_img_path "./test_movie_sucai_noise_001_no_ema" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 2514 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    # --mask_type i2v \
    # --enable_tiling 
