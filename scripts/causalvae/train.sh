# export https_proxy=http://127.0.0.1:8998
# export http_proxy=http://127.0.0.1:8998
unset https_proxy
unset http_proxy
export WANDB_PROJECT=causalvideovae_2.0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4

EXP_NAME=latent4_resume_2

torchrun \
    --nnodes=1 --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29600 \
    scripts/causalvae/train_causalvae.py \
    --exp_name ${EXP_NAME} \
    --model_config scripts/config.json \
    --video_path /storage/dataset/pexels/ \
    --eval_video_path /storage/dataset/pixabay_v2/folder_01 \
    --resolution 320 \
    --epochs 1000 \
    --num_frames 25 \
    --batch_size 1 \
    --disc_start 2000 \
    --save_ckpt_step 2000 \
    --eval_steps 500 \
    --eval_batch_size 1 \
    --eval_num_frames 33 \
    --eval_sample_rate 3 \
    --eval_subset_size 50 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --loss_type l1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --not_resume_training_process \
    --resume_from_checkpoint /storage/lcm/Causal-Video-VAE/results/latent8_3d-lr1.00e-05-bs1-rs320-sr2-fr25/checkpoint-14000.ckpt