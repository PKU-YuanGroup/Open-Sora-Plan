
PROMPT="GenAI527"
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/skip/deepnorm_skip/${PROMPT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch/skip/deepnorm_skip/checkpoint-309001/model_ema \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
    --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32