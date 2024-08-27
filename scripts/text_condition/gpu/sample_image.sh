export TASK_QUEUE_ENABLE=0

CUDA_VISIBLE_DEVICES=1 python opensora/sample/sample_t2v.py \
    --model_path /storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs32x8x8_vae8_anyx320x320_lr5e-5_snr5_noioff0.02_ema9999_sparse1d4_newdit_l_122_rope_mt5xxl_mj/checkpoint-218000/model_ema \
    --version 65x512x512 \
    --num_frames 1 \
    --height 320 \
    --width 320 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_2.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "WFVAE_DISTILL_FORMAL" \
    --save_img_path "sample_image_vae8_newdit_218k_320x320_test_sam" \
    --fps 24 \
    --guidance_scale 4.5 \
    --num_sampling_steps 28 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method DPMSolverMultistep \
    --model_type sparsedit 