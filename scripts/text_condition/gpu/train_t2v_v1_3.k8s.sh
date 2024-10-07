export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
export ENTITY="linbin"
export PROJECT="bs32x8x2_61x480p_lr1e-4_snr5_noioff0.02_opensora122_rope_mt5xxl_pandamovie_aes_mo_sucai_mo_speed1.2"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
# export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=25
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_RETRY_CNT=32
# export NCCL_ALGO=Tree

MAIN_PROCESS_IP=${1}
MAIN_PROCESS_PORT=${2}
NUM_MACHINES=${3}
NUM_PROCESSES=${4}
MACHINE_RANK=${5}

export TOKENIZERS_PARALLELISM=false

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example.k8s.yaml \
    --main_process_ip=${MAIN_PROCESS_IP} \
    --main_process_port=${MAIN_PROCESS_PORT} \
    --num_machines=${NUM_MACHINES} \
    --num_processes=${NUM_PROCESSES} \
    --machine_rank=${MACHINE_RANK} \
    opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V_v1_3-2B/122 \
    --text_encoder_name_1 google/mt5-xxl \
    --cache_dir "../../cache_dir/" \
    --dataset t2v \
    --data "scripts/train_data/merge_data.txt" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/latent8" \
    --sample_rate 1 \
    --num_frames 93 \
    --max_hxw 236544 \
    --min_hxw 102400 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1000 \
    --allow_tf32 \
    --model_max_length 512 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --resume_from_checkpoint="latest" \
    --skip_low_resolution \
    --speed_factor 1.0 \
    --ema_decay 0.9999 \
    --drop_short_ratio 0.0 \
    --pretrained "/storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs32x8x1_anyx93x640x640_fps16_lr1e-5_snr5_ema9999_sparse1d4_dit_l_mt5xxl_vpred_zerosnr/checkpoint-144000/model_ema/diffusion_pytorch_model.safetensors" \
    --hw_stride 32 \
    --sparse1d --sparse_n 4 \
    --train_fps 18 \
    --seed 1234 \
    --trained_data_global_step 0 \
    --group_data \
    --use_decord \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr \
    --output_dir="train_v1_3_any93x352x640_min320_vpre_nomotion_fps18" 
