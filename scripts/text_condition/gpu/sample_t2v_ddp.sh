# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m opensora.sample.sample_t2v_ddp \
    --model_path /storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs8x8x1_93x176x320_fps16_lr2e-5_snr5_noioff0.02_ema9999_sparse1d4_dit_l_mt5xxl_sucaiaes5_fromimgsparse1d4 \
    --version 65x512x512 \
    --num_frames 93 \
    --height 176 \
    --width 320 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D8_4x8x8 \
    --ae_path "/storage/dataset/new488dim8/last" \
    --save_img_path "./sample_video_dit_vae8_sparse1d4_anyx93x176x320_fps16_fromimagesparse1d4" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit"