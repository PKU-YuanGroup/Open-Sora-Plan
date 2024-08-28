export TASK_QUEUE_ENABLE=0
torchrun --nproc_per_node=1 opensora/sample/sample_t2v_on_npu.py \
    --model_path bs32x8x1_anyx93x320x320_fps16_lr1e-5_snr5_noioff0.02_ema9999_sparse1d4_dit_l_mt5xxl_alldata100m/model_ema \
    --num_frames 29 \
    --height 160 \
    --width 320 \
    --cache_dir "../cache_dir" \
    --text_encoder_name /home/image_data/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "WFVAE_DISTILL_FORMAL" \
    --save_img_path "./test_video" \
    --fps 24 \
    --guidance_scale 5.0 \
    --num_sampling_steps 24 \
    --sample_method PNDM \
    --model_type "sparsedit" \
    --motion_score 0.9 \