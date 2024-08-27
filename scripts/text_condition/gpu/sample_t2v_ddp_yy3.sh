# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \
CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --nproc_per_node 1 --master_port 29505 \
    -m opensora.sample.sample_t2v_ddp_vbench_gpt \
    --model_path /storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs32x8x1_anyx93x320x320_fps16_lr1e-5_snr5_noioff0.02_ema9999_sparse1d4_dit_l_mt5xxl_alldata100m/checkpoint-415000/model_ema \
    --version 65x512x512 \
    --num_frames 93 \
    --height 160 \
    --width 320 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt /storage/ongoing/refine_model/prompts_per_dimension/csv_gpt/human_action.csv \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/Causal-Video-VAE/results/WFVAE_DISTILL_FORMAL" \
    --save_img_path "/home/node-user/human_action_gpt" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "sparsedit" \
    --motion_score 0.9 \
    --seed 42 43 44 45 46
