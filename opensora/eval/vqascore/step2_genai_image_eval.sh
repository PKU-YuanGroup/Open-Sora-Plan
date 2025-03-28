PROMPT="GenAI527"
META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR} >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

PROMPT="GenAI527"
META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
CUDA_VISIBLE_DEVICES=3 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    

META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt
    
PROMPT="GenAI527"
META_DIR="opensora/eval/eval_prompts/${PROMPT}"
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
CUDA_VISIBLE_DEVICES=3 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR}  >> ${IMAGE_DIR}.txt