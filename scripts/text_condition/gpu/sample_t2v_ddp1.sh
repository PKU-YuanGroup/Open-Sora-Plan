# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29502 \
    -m opensora.sample.sample_t2v_ddp \
    --model_path bs32x8x2_anyx93x320x320_fps16_lr1e-5_snr5_ema9999_sparse1d4_dit_l_mt5xxl_alldata100m_vpred_zerosnr/checkpoint-3000/model \
    --version 65x512x512 \
    --num_frames 93 \
    --height 160 \
    --width 320 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt /storage/ongoing/new/7.19anyres/Open-Sora-Plan/examples/sora_refine.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/Causal-Video-VAE/results/WFVAE_DISTILL_FORMAL" \
    --save_img_path "./sample_video_dit_vae8_newmodel_anyx93x160x320_sora_m0.9_3k_s100_wf_xformer_refine_vpred_zerosnr_trailing" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "sparsedit" \
    --motion_score 0.9 \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr 