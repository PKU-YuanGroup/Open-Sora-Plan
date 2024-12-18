
PROMPT="PartiPrompts"
OUTPUT_DIR="opensora/eval/gen_img_for_human_pref/${PROMPT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nnodes=1 --nproc_per_node 7 --master_port 29513 \
     -m opensora.eval.imagereward.step1_gen_samples \
    --model_path /storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/12.11_mmdit13b_dense_rf_bs8192_lr1e-4_max1x384x384_min1x384x288_emaclip99_wd0_rms2layer/checkpoint-4506/model \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
    --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --ae_dtype fp16 \
    --weight_dtype fp16