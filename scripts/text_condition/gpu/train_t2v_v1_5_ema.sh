export WANDB_API_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
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
export TOKENIZERS_PARALLELISM=false
# export NCCL_ALGO=Tree

# MAIN_PROCESS_IP=${1}
# MAIN_PROCESS_PORT=${2}
# NUM_MACHINES=${3}
# NUM_PROCESSES=${4}
# MACHINE_RANK=${5}

# accelerate launch \
#     --config_file scripts/accelerate_configs/multi_node_example.k8s.yaml \
#     --main_process_ip=${MAIN_PROCESS_IP} \
#     --main_process_port=${MAIN_PROCESS_PORT} \
#     --num_machines=${NUM_MACHINES} \
#     --num_processes=${NUM_PROCESSES} \
#     --machine_rank=${MACHINE_RANK} \

# accelerate launch \
#     --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v_diffusers_ema.py \
    --train_deepspeed_config_file scripts/accelerate_configs/zero2.json \
    --eval_deepspeed_config_file scripts/accelerate_configs/zero3.json \
    --model OpenSoraT2V_v1_5-3B/122 \
    --text_encoder_name_1 google/t5-v1_1-xl \
    --cache_dir "../../cache_dir/" \
    --text_encoder_name_2 laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data scripts/train_data/image_data_debug.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_hxw 65536 \
    --min_hxw 36864 \
    --force_5_ratio \
    --gradient_checkpointing \
    --train_batch_size=8 \
    --dataloader_num_workers 16 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant_with_warmup" \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1000 \
    --allow_tf32 \
    --model_max_length 512 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --resume_from_checkpoint="latest" \
    --ema_update_freq 1 \
    --ema_decay 0.9999 \
    --drop_short_ratio 0.0 \
    --hw_stride 16 \
    --train_fps 16 \
    --seed 1234 \
    --group_data \
    --use_decord \
    --output_dir="debug_ema_acce1.0.1" \
    --vae_fp32 \
    --rf_scheduler \
    --proj_name "debug_ema" \
    --post_to_device \
    --log_name debug_ema_acce1.0.1 \
    --skip_abnorml_step --ema_decay_grad_clipping 0.99