# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m opensora.sample.sample_t2v_ddp \
    --model_path /storage/ongoing/new/Open-Sora-Plan/bs32x8x1_29x720p_lr1e-4_snr5_noioff0.02_ema999_opensora122_rope_fp32_mt5xxl_pandamovie_aes_mo \
    --version 65x512x512 \
    --num_frames 29 \
    --height 720 \
    --width 1280 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_1.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --save_img_path "./sample_video_bs256_ddp_29_29x720p_cfg7.5_step100_neg_pos" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --sample_method PNDM \
    --model_type "dit"