export PROJECT_NAME="test_4_node_on_istock"
export PROJECT_EXP_NAME="test"
export PROJECT_DIR="/work/share/checkpoint/gyy/osp/$PROJECT_NAME"

SAMPLER_FILE_PATH="$PROJECT_DIR/global_step_for_sampler.txt"

run_bash() {
    local file="$1"
    local command="$2"
    # 刚开始固定运行一次
    bash -c "$command"  # 运行传入的命令
    while [ -f "$file" ]; do
        echo "$file 文件存在，循环继续..."
        sleep 5s  # 防止高CPU占用
        bash -c "$command"  # 运行传入的命令
    done
    echo "$file 文件不存在，循环结束..."
    sleep 5s
}

# 依次执行不同的数据文件
for i in {0..9}; do
    export MM_DATA="./examples/opensoraplan1.5/data0$i.json" 
    run_bash "$SAMPLER_FILE_PATH" "examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh"
done