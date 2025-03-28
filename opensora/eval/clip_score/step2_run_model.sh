
cd /storage/ongoing/12.13/t2i/Open-Sora-Plan
conda activate t2i
CUDA_VISIBLE_DEVICES=1 python -m opensora.eval.clip_score.step2_run_model \
--model_path "/storage/ysh/Ckpts/openai/clip-vit-base-patch16" \
--image_dir "/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_s24_nocfg/umt5/base/COCO2017" \
--prompt_type COCO2017

cd /storage/ongoing/12.13/t2i/Open-Sora-Plan
conda activate t2i
CUDA_VISIBLE_DEVICES=1 python -m opensora.eval.clip_score.step2_run_model \
--model_path "/storage/ysh/Ckpts/openai/clip-vit-base-patch16" \
--image_dir "/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_s24_nocfg/umt5/base/ImageNet" \
--prompt_type ImageNet
