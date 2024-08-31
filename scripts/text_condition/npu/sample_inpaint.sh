export TASK_QUEUE_ENABLE=0
torchrun --nnodes=1 --nproc_per_node=2 --master_port 29502 \
    -m opensora.sample.sample_inpaint_ddp_on_npu  \
    --model_path /home/image_data/gyy/Open-Sora-Plan/test_sparse_inpaint/checkpoint-10000/model \
    --num_frames 93 \
    --height 320 \
    --width 160 \
    --cache_dir "../cache_dir" \
    --text_encoder_name /home/image_data/mt5-xxl \
    --text_prompt /home/image_data/test_prompt.txt \
    --conditional_images_path /home/image_data/test_cond_imgs_path.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/image_data/lb/Open-Sora-Plan/WFVAE_DISTILL_FORMAL" \
    --save_img_path "./test_video_new" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --sample_method PNDM \
    --motion_score 0.9 \