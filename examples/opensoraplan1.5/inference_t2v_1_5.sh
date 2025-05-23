source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_LAUNCH_BLOCKING=1
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

MASTER_ADDR=localhost
MASTER_PORT=29200
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

# MINDSPEED_PATH="./MindSpeed/"
# export PYTHONPATH=${MINDSPEED_PATH}:$PYTHONPATH

export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

MM_MODEL="examples/opensoraplan1.5/inference_t2v_model1_5.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
MM_ARGS="
 --mm-model $MM_MODEL
"

SORA_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-layers 32 \
    --hidden-size 3072 \
    --num-attention-heads 16 \
    --seq-length 1024\
    --max-position-embeddings 1024 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 2e-5 \
    --min-lr 2e-5 \
    --train-iters 5010 \
    --weight-decay 0 \
    --clip-grad 1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --sequence-parallel \
    --distributed-timeout-minutes 20 \
    --seed 1235 \
    --optimizer-selection fused_torch_adamw \
"

torchrun $DISTRIBUTED_ARGS  inference_sora.py  $MM_ARGS $SORA_ARGS 2>&1 | tee logs/inference_test.log
