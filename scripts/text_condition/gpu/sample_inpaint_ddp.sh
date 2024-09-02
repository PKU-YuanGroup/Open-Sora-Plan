# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 2 --master_port 29502 \
    -m opensora.sample.sample_inpaint_ddp \
    --model_path /storage/gyy/hw/Open-Sora-Plan/test_sparse_inpaint/checkpoint-84000/model \
    --num_frames 93 \
    --height 160 \
    --width 320 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt /storage/gyy/hw/Open-Sora-Plan/test_prompt.txt \
    --conditional_images_path /storage/gyy/hw/Open-Sora-Plan/test_cond_imgs_path.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/Causal-Video-VAE/results/WFVAE_DISTILL_FORMAL" \
    --save_img_path "./sample_test_inpaint_sparse" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "sparsedit" \
    --motion_score 0.9 \
    --seed 1234