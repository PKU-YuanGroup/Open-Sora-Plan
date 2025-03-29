
PROMPT="GenEval"
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4/checkpoint-309000/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4-9b-chat \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4v/checkpoint-309000/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4v-9b \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/mixnorm/checkpoint-309125/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/norm/deepnorm_shrink/checkpoint-309001/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/postnorm/checkpoint-309125/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/prenorm/checkpoint-309125/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16



PROMPT="GenEval"


     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/qwenvl2/checkpoint-309000/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/Qwen2-VL-7B-Instruct \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/sandwich/checkpoint-309125/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/deepnorm_skip/checkpoint-309001/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/postnorm_skip/checkpoint-309001/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_normskip/checkpoint-309000/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_skip/checkpoint-309000/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
     --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     
PROMPT="GenEval"
OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29213 \
     -m opensora.eval.geneval.generate_samples \
     --prompt_path opensora/eval/geneval/evaluation_metadata.jsonl \
     --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/umt5/checkpoint-309000/model_ema \
     --output_dir ${OUTPUT_DIR} \
     --version t2i \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl" \
     --num_sampling_steps 24 \
     --ae WFVAE2Model_D32_1x8x8 \
     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
     --ae_dtype fp16 \
     --weight_dtype fp16

     