
PROMPT="GenAI527"
OUTPUT_DIR="opensora/eval/gen_img_for_human_pref_6b_suv_24k/${PROMPT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch/sandwich/checkpoint-162000/model_ema \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl" \
    --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32