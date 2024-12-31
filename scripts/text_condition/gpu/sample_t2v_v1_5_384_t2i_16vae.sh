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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.sample.sample \
    --model_path t2i_ablation_arch/vae/prenorm_1616/checkpoint-54000/model_ema \
    --version t2i \
    --num_frames 1 \
    --height 256 \
    --width 256 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl" \
    --text_prompt examples/prompt.txt \
    --ae WFVAE2Model_D128_1x16x16 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_11616128_slim" \
    --save_img_path "./256x256_cfg7.0_t2i_16vae_54k_ema" \
    --fps 18 \
    --guidance_scale 7.0 \
    --guidance_rescale 0.7 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method OpenSoraFlowMatchEuler \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --use_linear_quadratic_schedule \
    --use_pos_neg_prompt \
    --ae_dtype fp16 \
    --weight_dtype fp16