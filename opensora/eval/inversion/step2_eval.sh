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

accelerate launch --main_process_port 29500 --num_processes 1 \
    opensora/eval/inversion/step2_eval.py \
    --data_file /storage/dataset/inversion_data/in-domain/data/flower102/test.json \
    --data_root "/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_inversion_nocondi/mixnorm/flowers102/test" \
    --num_workers 8 \
    --num_inverse_steps 40 80 90 100 \
    --num_samples 100