#!/bin/bash

cd /storage/ongoing/12.13/t2i/Open-Sora-Plan
conda activate t2i

PROMPTS=('ImageNet' 'COCO2017')

for PROMPT in "${PROMPTS[@]}"; do
    ablation_type="umt5"

    OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_s24_nocfg/${ablation_type}/base/${PROMPT}"
    RESULT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch_res_s24_nocfg/${ablation_type}/base"
    META_DIR="opensora/eval/eval_prompts/${PROMPT}"
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${RESULT_DIR}
    RESULT_FILE_PATH="${RESULT_DIR}/${PROMPT}.txt"
    if [ -f "$RESULT_FILE_PATH" ]; then
        echo "File $RESULT_FILE_PATH exists. Skipping..."
        continue
    fi

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
        -m opensora.eval.step1_gen_samples \
        --version t2i \
        --model_path "/storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch_/umt5/checkpoint-309000/model_ema" \
        --output_dir ${OUTPUT_DIR} \
        --prompt_type ${PROMPT} \
        --text_encoder_name_1 "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl" \
        --ae WFVAE2Model_D32_1x8x8 \
        --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
        --ae_dtype fp16 \
        --weight_dtype fp16 \
        --allow_tf32 \
        --num_sampling_steps 24 \
        --guidance_rescale 0.0 \
        --guidance_scale 1.0
done
