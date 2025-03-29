
# PROMPT="DrawBench"
# IMAGE_DIR="opensora/eval/gen_img_for_human_pref_6b/${PROMPT}"

# CUDA_VISIBLE_DEVICES=0 python -m opensora.eval.vqascore.step2_run_model \
#     --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
#     --image_dir ${IMAGE_DIR} \
#     --prompt_type ${PROMPT} 



# PROMPTS=('DALLE3' 'DOCCI-Test-Pivots' 'DrawBench' 'Gecko-Rel' 'PartiPrompts')
# PROMPTS=('DALLE3' 'DrawBench')
# PROMPTS=('DOCCI-Test-Pivots' 'PartiPrompts')
PROMPTS=('Gecko-Rel')

for PROMPT in "${PROMPTS[@]}"; do
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
    CUDA_VISIBLE_DEVICES=0 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
    CUDA_VISIBLE_DEVICES=1 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
    CUDA_VISIBLE_DEVICES=2 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
    CUDA_VISIBLE_DEVICES=3 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
    CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
    CUDA_VISIBLE_DEVICES=5 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
    CUDA_VISIBLE_DEVICES=6 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
    CUDA_VISIBLE_DEVICES=7 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
    CUDA_VISIBLE_DEVICES=0 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
    CUDA_VISIBLE_DEVICES=1 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
    CUDA_VISIBLE_DEVICES=2 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
    CUDA_VISIBLE_DEVICES=3 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt & \
    IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
    CUDA_VISIBLE_DEVICES=4 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} >> ${IMAGE_DIR}.txt
done