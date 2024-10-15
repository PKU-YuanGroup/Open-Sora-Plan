source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TASK_QUEUE_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/home/image_data/yancen/MindSpeed-MM:$PYTHONPATH
MASTER_ADDR=localhost
MASTER_PORT=12875
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

export use_debug=0



TP=4
PP=1
CP=2
MBS=1
GBS=1
MM_MODEL="examples/opensoraplan1.3/inference_t2v_model1_3.json"

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
    --hidden-size 2304 \
    --num-attention-heads 24 \
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
    --bf16 \
"

torchrun $DISTRIBUTED_ARGS  inference_sora.py  $MM_ARGS $SORA_ARGS
