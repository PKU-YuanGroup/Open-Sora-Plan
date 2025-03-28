#!/bin/bash

model_path=$1
ablation_type=$2

echo ${model_path}
echo ${ablation_type}

num_layers=(12 1 12)
PROMPT="GenAI527"

OUTPUT_DIR="t2i_ablation_arch_gen/${ablation_type}/base/${PROMPT}"
RESULT_DIR="t2i_ablation_arch_res/${ablation_type}/base"
META_DIR="opensora/eval/eval_prompts/${PROMPT}"
mkdir -p ${RESULT_DIR}
RESULT_FILE_PATH="${RESULT_DIR}/${PROMPT}.txt"
if [ -f "$RESULT_FILE_PATH" ]; then
    echo "File $RESULT_FILE_PATH exists. Skipping..."
    exit
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.eval.step1_gen_samples \
    --version t2i \
    --model_path ${model_path} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl" \
    --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32

CUDA_VISIBLE_DEVICES=0 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/clip-flant5-xxl" \
    --image_dir ${OUTPUT_DIR} \
    --meta_dir ${META_DIR} >> ${RESULT_FILE_PATH}