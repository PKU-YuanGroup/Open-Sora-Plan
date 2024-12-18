
PROMPT="PartiPrompts"
IMAGE_DIR="opensora/eval/gen_img_for_human_pref/${PROMPT}"

CUDA_VISIBLE_DEVICES=0 python -m opensora.eval.hpsv2.step2_run_model \
    --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/HPSv2 \
    --tokenizer_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --image_dir ${IMAGE_DIR} \
    --prompt_type ${PROMPT} 