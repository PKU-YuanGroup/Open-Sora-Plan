
PROMPT="GenAI527"
OUTPUT_DIR="opensora/eval/gen_img_for_human_pref_6b_suv_24k/${PROMPT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/dataset/ospv1_5_image_ckpt/suv_24k/model_ema \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
    --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32