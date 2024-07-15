# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m opensora.sample.sample_t2v_ddp \
    --model_path /storage/ongoing/new/Open-Sora-Plan/bs36xsp2_29x720p_lr1e-4_snr5_noioff0.02_ema999_opensora122_rope_fp32_mt5xxl_pandamovie_aes_mo_sucai_mo_speed1.1 \
    --version 65x512x512 \
    --num_frames 29 \
    --height 720 \
    --width 1280 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_1.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --save_img_path "./sample_video_ddp_29_29x720p_cfg5.0_step50_neg_pos_dmp" \
    --fps 24 \
    --guidance_scale 5.0 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --sample_method DPMSolverMultistep \
    --model_type "dit"