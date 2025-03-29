
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="/storage/lcm/image_branch/Open-Sora-Plan/opensora/eval/geneval/detector"
PROMPT="GenEval"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
python opensora/eval/geneval/evaluate_images.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}/geneval_results.jsonl \
    --model-path ${MODEL_PATH}
