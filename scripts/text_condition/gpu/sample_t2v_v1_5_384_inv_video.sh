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

CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node 1 --master_port 29513 \
    -m opensora.sample.sample \
    --model_path 11.15_mmdit13b_dense_rf_bs8192_lr1e-4_max105x384x384_min29x384x288_emaclip99/checkpoint-24662/model_ema \
    --version v1_5 \
    --num_frames 33 \
    --height 288 \
    --width 512 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
    --text_encoder_name_2 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --text_prompt examples/inv_prompt_video.txt \
    --inversion_path examples/inv_video.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --save_img_path "./rf_33x288x512_v1_5_13b_cfg7.0_s100_inv80_24k_5k_inv_noposneg" \
    --fps 18 \
    --guidance_scale 7.0 \
    --num_sampling_steps 100 \
    --num_inverse_steps 80 \
    --max_sequence_length 512 \
    --sample_method FlowMatchEulerDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --v1_5_scheduler 