
# PROMPT="DrawBench"
# IMAGE_DIR="opensora/eval/gen_img_for_human_pref_6b/${PROMPT}"

# CUDA_VISIBLE_DEVICES=0 python -m opensora.eval.vqascore.step2_run_model \
#     --model_path "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/clip-flant5-xxl" \
#     --image_dir ${IMAGE_DIR} \
#     --prompt_type ${PROMPT} 



PROMPTS=('DALLE3' 'DOCCI-Test-Pivots' 'DrawBench' 'Gecko-Rel' 'PartiPrompts')

for PROMPT in "${PROMPTS[@]}"; do
    IMAGE_DIR="opensora/eval/gen_img_for_human_pref_6b_suv_24k/${PROMPT}"
    echo ${PROMPT}
    CUDA_VISIBLE_DEVICES=6 python -m opensora.eval.vqascore.step2_run_model \
    --model_path "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} 
done