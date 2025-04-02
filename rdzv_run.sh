export PROJECT_NAME="105x288x512_node48_tp4_bs4_gc2_lr1e-4_wd1e-4"
# export PROJECT_NAME="test_4node_final"
export PROJECT_EXP_NAME="all_data"
export PROJECT_DIR="/work/share1/checkpoint/$PROJECT_NAME"

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

# # 依次执行不同的数据文件
# export MM_DATA="./examples/opensoraplan1.5/data00.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data01.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh
export MM_DATA="./examples/opensoraplan1.5/data02.json" 
run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

export MM_DATA="./examples/opensoraplan1.5/data03.json" 
run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data04.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data05.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data06.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data07.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data08.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data09.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data10.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data11.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data12.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data13.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data14.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data15.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data16.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data17.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data18.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh

# export MM_DATA="./examples/opensoraplan1.5/data19.json" 
# run_bash "$SAMPLER_FILE_PATH" examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh