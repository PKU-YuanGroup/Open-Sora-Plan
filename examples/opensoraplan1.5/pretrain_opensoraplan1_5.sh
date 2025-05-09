#!/bin/bash
wandb login 720d886d8c437c2142c88056a1eab8ef78d64a1f
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=0
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

MINDSPEED_PATH="./MindSpeed/"
export PYTHONPATH=${MINDSPEED_PATH}:$PYTHONPATH

export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
# export GLOO_SOCKET_IFNAME=enp67s0f0

# NNODES=${PET_NNODES}
# NODE_RANK=${PET_NODE_RANK}
# MASTER_ADDR=${PET_MASTER_ADDR}

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost

MASTER_NODE_ID=0
GPUS_PER_NODE=8
MASTER_PORT=12345
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=8
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=1
GBS=$(($WORLD_SIZE*$GRAD_ACC_STEP*$MBS/$CP/$TP))

MM_DATA="./examples/opensoraplan1.5/data_test.json"
MM_MODEL="./examples/opensoraplan1.5/model_test.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"

PROJECT_DIR="./test_ckpt/test_cann_8_0_1_1_720p"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
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
    --weight-decay 1e-4 \
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
    --accumulate-allreduce-grads-in-fp32 
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
    --save $PROJECT_DIR
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 10 \
    --eval-iters 10 \
"

WANDB_ARGS="
    --wandb-project test_1_node \
    --wandb-exp-name test_1_node \
    --wandb-save-dir . \
    --tensorboard-log-interval 1 \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    $WANDB_ARGS \
    --distributed-backend nccl 2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
FPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME, Average FPS: $FPS"
