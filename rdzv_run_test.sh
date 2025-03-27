# export PROJECT_NAME="105x288x512_node48_tp4_bs4_gc2_lr1e-4_wd1e-4"
export PROJECT_NAME="test_rdzv"
export PROJECT_EXP_NAME="test"
export PROJECT_DIR="/work/share/checkpoint/gyy/osp/$PROJECT_NAME"

SAMPLER_FILE_PATH="$PROJECT_DIR/global_step_for_sampler.txt"

run_bash() {
    local file="$1"
    local command="$2"
    # 刚开始固定运行一次
    bash "$command"  # 运行传入的命令
    while [ -f "$file" ]; do
        echo "$file 文件存在，循环继续..."
        bash "$command"  # 运行传入的命令
    done
    echo "$file 文件不存在，循环结束..."
}


export MM_DATA="./examples/opensoraplan1.5/dataset_test.json" 
run_bash "$SAMPLER_FILE_PATH" "examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh"