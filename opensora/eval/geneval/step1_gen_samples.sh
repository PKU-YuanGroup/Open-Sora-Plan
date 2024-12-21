
OUTPUT_DIR="opensora/eval/geneval/geneval_outputs"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/12.11_mmdit13b_dense_rf_bs8192_lr1e-4_max1x384x384_min1x384x288_emaclip99_wd0_rms2layer/checkpoint-4506/model \
     --output_dir ${OUTPUT_DIR} \
     --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
     --num_sampling_steps 32 \
     --text_encoder_name_3 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
     --ae WFVAEModel_D32_8x8x8 \
     --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
     --ae_dtype fp32 \
     --weight_dtype fp32