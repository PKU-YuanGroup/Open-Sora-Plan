
META_DIR="opensora/eval/eval_prompts/GenAI-Image-527"
IMAGE_DIR="opensora/eval/gen_img_for_human_pref/GenAI"

CUDA_VISIBLE_DEVICES=1 python -m opensora.eval.vqascore.step2_genai_image_eval \
    --model_path "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/clip-flant5-xxl" \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR} 