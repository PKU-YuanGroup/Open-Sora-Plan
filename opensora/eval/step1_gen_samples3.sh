


PROMPT="PartiPrompts"
export NGPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4-9b-chat \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4v/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4v-9b \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/mixnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/norm/deepnorm_shrink/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/postnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/prenorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32



OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/qwenvl2/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/Qwen2-VL-7B-Instruct \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/sandwich/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/deepnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/postnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_normskip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_skip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/umt5/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl" \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32








PROMPT="DOCCI-Test-Pivots"
export NGPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4-9b-chat \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4v/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4v-9b \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/mixnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/norm/deepnorm_shrink/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/postnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/prenorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32



OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/qwenvl2/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/Qwen2-VL-7B-Instruct \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/sandwich/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/deepnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/postnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_normskip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_skip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/umt5/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl" \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32








PROMPT="DrawBench"
export NGPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4-9b-chat \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4v/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4v-9b \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/mixnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/norm/deepnorm_shrink/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/postnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/prenorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32



OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/qwenvl2/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/Qwen2-VL-7B-Instruct \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/sandwich/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/deepnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/postnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_normskip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_skip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/umt5/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl" \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32








PROMPT="DALLE3"
export NGPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4-9b-chat \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4v/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4v-9b \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/mixnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/norm/deepnorm_shrink/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/postnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/prenorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32



OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/qwenvl2/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/Qwen2-VL-7B-Instruct \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/sandwich/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/deepnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/postnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_normskip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_skip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/umt5/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl" \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32







PROMPT="Gecko-Rel"
export NGPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4-9b-chat \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/glm4v/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/glm-4v-9b \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/mixnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/norm/deepnorm_shrink/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/postnorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/prenorm/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32



OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/qwenvl2/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/Qwen2-VL-7B-Instruct \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/sandwich/checkpoint-309125/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/deepnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/postnorm_skip/checkpoint-309001/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_normskip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/skip/prenorm_skip/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 /storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32




OUTPUT_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
torchrun --nnodes=1 --nproc_per_node ${NGPU} --master_port 29513 \
     -m opensora.eval.step1_gen_samples \
    --model_path /storage/ongoing/12.13/t2i/t2i_ablation_arch/umt5/checkpoint-309000/model_ema \
    --version t2i \
    --output_dir ${OUTPUT_DIR} \
    --prompt_type ${PROMPT} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl" \
     --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --ae_dtype fp16 \
    --weight_dtype fp16 \
    --allow_tf32





