accelerate launch --main_process_port 29500 --num_processes 1 --mixed_precision fp16 \
    opensora/eval/inversion/eval.py \
    --ae_path /storage/lcm/WF-VAE/results/Middle888 \
    --model_path /storage/dataset/ospv1_5_image_ckpt/wd1e-1_last_ckpt/model_ema/ \
    --data_txt /storage/lcm/image_branch/Open-Sora-Plan/opensora/eval/inversion/image_data.txt \
    --text_encoder_name_1 /storage/cache_dir/t5-v1_1-xl \
    --text_encoder_name_3 /storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --resolution 384 \
    --num_workers 4 \
    --num_inverse_steps 10 30 50 70 \
    --num_inference_steps 100 \
    --guidance_scale 7.0 \
    --num_samples 5