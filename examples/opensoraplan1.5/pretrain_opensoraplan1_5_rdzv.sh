#!/bin/bash
set -e
wandb login 720d886d8c437c2142c88056a1eab8ef78d64a1f
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export ACL_DEVICE_SYNC_TIMEOUT=3600
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export GPU_NUM_PER_NODE=8

MINDSPEED_PATH="./MindSpeed/"
export PYTHONPATH=${MINDSPEED_PATH}:$PYTHONPATH

export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/aarch64-linux/devlib/libascend_hal.so:$LD_LIBRARY_PATH
# export GLOO_SOCKET_IFNAME=enp67s0f0

# 平台断点续训，增加pipe fail捕获退出码
set -o pipefail

# # 平台参数解析，使用冒号作为分隔符分割字符串
# IFS=':' read -r -a array <<< "${PET_RDZV_ENDPOINT}"
# main_process_ip=${array[0]} # 主节点ip
# main_process_port=${array[1]} # 主节点端口

# IFS=':' read -r first_value second_value <<< "$PET_NNODES"
# NNODE=${first_value//[^0-9]/} # 节点总数目
# NUM_NPUS=$(($NNODE * $GPU_NUM_PER_NODE)) # 总卡数

# # 使用正则表达式提取最后一个-后面的数字
# if [[ $o$POD_NAME =~ -([0-9]+)$ ]]; then
#     RANK=${BASH_REMATCH[1]}
#     # 将提取的数字转换为整数
#     RANK=${RANK//[^0-9]/}
#     echo "The rank is: $RANK"
# else
#     echo "No match found"
# fi
# echo "----------------"
# echo $NUM_NPUS
# echo $NNODE

TP=4
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=4
NUM_NPUS=$(($PET_NNODES * $GPU_NUM_PER_NODE))
GBS=$(($NUM_NPUS*$GRAD_ACC_STEP*$MBS/$CP/$TP))

MM_MODEL="./examples/opensoraplan1.5/model_opensoraplan1_5.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
MM_DATA="./examples/opensoraplan1.5/data00.json"

# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPU_NUM_PER_NODE \
#     --nnodes ${PET_NNODES} \
#     --rdzv_backend=${PET_RDZV_BACKEND} \
#     --rdzv_endpoint=${PET_RDZV_ENDPOINT} \
#     --rdzv_id=${PET_RDZV_ID} \
#     --max_restarts=25 \
#     --rdzv_conf=timeout=7200,read_timeout=7200 \
# "

DISTRIBUTED_ARGS="
    --nproc_per_node=${GPU_NUM_PER_NODE} \
    --nnodes=${PET_NNODES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT}
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-layers 32 \
    --hidden-size 3072 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --bf16 \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-15 \
    --lr-decay-style constant \
    --weight-decay 1e-2 \
    --lr-warmup-init 1e-5 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters 100000000 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 0 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --qk-layernorm \
    --sequence-parallel \
    --optimizer-selection fused_ema_adamw \
    --seed 1024 \
    --data-parallel-random-init \
    --use-ema \
    --load $PROJECT_DIR \
    --fp32-residual-connection \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --distributed-timeout-minutes 30
"

    # --no-load-optim \
    # --no-load-rng \
    # --no-save-optim \
    # --no-save-rng \


MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL \
    --model_custom_precision \
    --clip_grad_ema_decay 0.99
"

OUTPUT_ARGS="
    --save $PROJECT_DIR \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10 \
    --eval-iters 10 \
"

WANDB_ARGS="
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $PROJECT_EXP_NAME \
    --wandb-save-dir $PROJECT_DIR \
    --tensorboard-log-interval 1 \
"

logfile=${PROJECT_EXP_NAME}_node${RANK}_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs/$PROJECT_NAME
torchrun $DISTRIBUTED_ARGS pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    $WANDB_ARGS \
    --distributed-backend nccl 2>&1 | tee logs/$PROJECT_NAME/train_${logfile}.log

chmod 440 logs/$PROJECT_NAME/train_${logfile}.log