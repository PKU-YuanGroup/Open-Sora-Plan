#!/bin/bash

# Define the num_layers array
num_layers=(12 1 12)

# Outer loop for tr_stage
for tr_stage in "${!num_layers[@]}"; do
    tr_layers=${num_layers[$tr_stage]}
    for ((tr_layer=0; tr_layer<tr_layers; tr_layer++)); do
        echo "/storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch_del/sandwich/del_s${tr_stage}_l${tr_layer}"
        # Add your commands here
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29512 \
            -m opensora.eval.dpgbench.step1_gen_samples \
            --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch_del/sandwich/del_s${tr_stage}_l${tr_layer} \
            --text_encoder_name_1 "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl" \
            --ae WFVAE2Model_D32_1x8x8 \
            --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
            --ae_dtype fp16 \
            --weight_dtype fp16 \
            --prompt_path /storage/hxy/t2i/osp/Open-Sora-Plan/opensora/eval/eval_prompts/DPGbench/dpgbench_prompts.json \
            --result_path /storage/hxy/t2i/results/dpgbench/results/sandwich/del_s${tr_stage}_l${tr_layer} \
            --version t2i
    done
done




# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m opensora.eval.dpgbench.step1_gen_samples \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29512 \
#     -m opensora.eval.dpgbench.step1_gen_samples \
#     --model_path /storage/dataset/ospv1_5_image_ckpt/mmdit_lask_ckpt/model_ema \
#     --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
#     --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
#     --ae WFVAEModel_D32_8x8x8 \
#     --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
#     --ae_dtype fp16 \
#     --weight_dtype fp16 \
#     --prompt_path /storage/hxy/t2i/osp/Open-Sora-Plan/opensora/eval/eval_prompts/DPGbench/prompts \
#     --result_path /storage/hxy/t2i/results/dpgbench/results/mmdit_lask_ckpt \
    # --world_size 8 \
    # --model_path /storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/12.11_mmdit13b_dense_rf_bs8192_lr1e-4_max1x384x384_min1x384x288_emaclip99_wd0_rms2layer/checkpoint-4506/model \