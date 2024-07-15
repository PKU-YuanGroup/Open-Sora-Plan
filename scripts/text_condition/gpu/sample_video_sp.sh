# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \

torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m opensora.sample.sample_t2v_sp \
    --model_path /storage/dataset/hw29/model_ema \
    --version 65x512x512 \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_2.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --save_img_path "./sample_video_sp_29x720p_cfg10.0_step50_neg_pos_beng" \
    --fps 24 \
    --guidance_scale 2.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --sample_method DDIM \
    --model_type "dit" 