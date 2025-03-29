
# PROMPT="PartiPrompts"
# IMAGE_DIR="opensora/eval/gen_img_for_human_pref/${PROMPT}"

# CUDA_VISIBLE_DEVICES=0 python -m opensora.eval.imagereward.step2_run_model \
#     --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/ImageReward \
#     --tokenizer_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/bert-base-uncased \
#     --image_dir ${IMAGE_DIR} \
#     --prompt_type ${PROMPT} 



PROMPTS=('DALLE3' 'DOCCI-Test-Pivots' 'DrawBench' 'Gecko-Rel' 'PartiPrompts')

for PROMPT in "${PROMPTS[@]}"; do
    IMAGE_DIR="opensora/eval/gen_img_for_human_pref_6b_suv_24k/${PROMPT}"
    echo ${PROMPT}
    CUDA_VISIBLE_DEVICES=3 python -m opensora.eval.imagereward.step2_run_model \
    --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/ImageReward \
    --tokenizer_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/bert-base-uncased \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} 
done
