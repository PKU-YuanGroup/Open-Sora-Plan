
OUTPUT_DIR="opensora/eval/geneval/geneval_outputs"

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nnodes=1 --nproc_per_node 5 --master_port 29513 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/dataset/ospv1_5_image_ckpt/wd1e-1_last_ckpt/model_ema/ \
     --output_dir ${OUTPUT_DIR} \
     --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
     --num_sampling_steps 32 \
     --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
     --ae WFVAEModel_D32_8x8x8 \
     --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
     --ae_dtype fp16 \
     --weight_dtype fp16