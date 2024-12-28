
# PROMPT="DrawBench"
# IMAGE_DIR="opensora/eval/gen_img_for_human_pref_6b/${PROMPT}"

# CUDA_VISIBLE_DEVICES=0 python opensora/eval/mps/step2_run_model.py \
#     --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/MPS \
#     --tokenizer_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/CLIP-ViT-H-14-laion2B-s32B-b79K \
#     --image_dir ${IMAGE_DIR} \
#     --prompt_type ${PROMPT} 


PROMPTS=('DALLE3' 'DOCCI-Test-Pivots' 'DrawBench' 'Gecko-Rel' 'PartiPrompts')

for PROMPT in "${PROMPTS[@]}"; do
    IMAGE_DIR="opensora/eval/gen_img_for_human_pref_6b_suv_24k/${PROMPT}"
    echo ${PROMPT}
    CUDA_VISIBLE_DEVICES=5 python opensora/eval/mps/step2_run_model.py \
        --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/MPS \
        --tokenizer_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/CLIP-ViT-H-14-laion2B-s32B-b79K \
        --image_dir ${IMAGE_DIR} \
        --prompt_type ${PROMPT}
done
