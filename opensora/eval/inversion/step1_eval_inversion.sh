export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=25
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_RETRY_CNT=32
# export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

# accelerate launch --main_process_port 29500 --num_processes 8 --mixed_precision fp16 \
#     opensora/eval/inversion/step1_gen_samples.py \
#     --ae WFVAE2Model_D32_1x8x8 \
#     --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
#     --model_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch/sandwich/checkpoint-162000/model_ema \
#     --data_path /storage/dataset/inversion_data/in-domain/data/flower102/val.json \
#     --data_root /storage/dataset/inversion_data/in-domain/data/flower102 \
#     --text_encoder_name_1 "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl" \
#     --height 256 \
#     --width 256 \
#     --num_workers 8 \
#     --num_inverse_steps 40 80 90 100 \
#     --num_inference_steps 100 \
#     --guidance_scale 1.0 \
#     --no_condition \
#     --save_img_path "/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_inversion_nocondi/flowers102"



ablation_type="postnorm"
dataset_type="flowers102"
eval_root="/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_inversion_nocondi"
data_root="/storage/dataset/inversion_data/in-domain/data/flower102"
data_subset="test"
data_path="/storage/dataset/inversion_data/in-domain/data/flower102/${data_subset}.json"
model_path="/storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch/postnorm/checkpoint-309125/model_ema"
accelerate launch --main_process_port 29500 --num_processes 8 --mixed_precision fp16 \
    opensora/eval/inversion/step1_gen_samples.py \
    --ae WFVAE2Model_D32_1x8x8 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_18832_slim" \
    --model_path ${model_path} \
    --data_path ${data_path} \
    --data_root ${data_root} \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl" \
    --height 256 \
    --width 256 \
    --num_workers 8 \
    --num_inverse_steps 40 80 90 100 \
    --num_inference_steps 100 \
    --guidance_scale 1.0 \
    --no_condition \
    --save_img_path "${eval_root}/${ablation_type}/${dataset_type}"