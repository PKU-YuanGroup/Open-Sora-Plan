

CUDA_VISIBLE_DEVICES=2 python -m opensora.eval.gen_samples \
    --model_path /storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/12.11_mmdit13b_dense_rf_bs8192_lr1e-4_max1x384x384_min1x384x288_emaclip99_wd0_rms2layer/checkpoint-4506/model \
    --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
    --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --ae_dtype fp16 \
    --weight_dtype fp16