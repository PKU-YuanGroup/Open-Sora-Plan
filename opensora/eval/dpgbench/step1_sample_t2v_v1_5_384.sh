

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m opensora.eval.dpgbench.step1_gen_samples \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29512 \
    -m opensora.eval.dpgbench.step1_gen_samples \
    --model_path /storage/dataset/ospv1_5_image_ckpt/mmdit_lask_ckpt/model_ema \
    --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
    --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --prompt_path /storage/hxy/t2i/osp/Open-Sora-Plan/opensora/eval/eval_prompts/DPGbench/prompts \
    --result_path /storage/hxy/t2i/results/dpgbench/results/mmdit_lask_ckpt \
    # --world_size 8 \
    # --model_path /storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/12.11_mmdit13b_dense_rf_bs8192_lr1e-4_max1x384x384_min1x384x288_emaclip99_wd0_rms2layer/checkpoint-4506/model \