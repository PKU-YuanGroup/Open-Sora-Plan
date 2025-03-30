export PROJECT_NAME="105x288x512_node48_tp4_bs4_gc2_lr1e-4_wd1e-4"
# export PROJECT_NAME="test_4node_final"
export PROJECT_EXP_NAME="all_data"
export PROJECT_DIR="/work/share1/checkpoint/$PROJECT_NAME"

SAMPLER_FILE_PATH="$PROJECT_DIR/global_step_for_sampler.txt"

bash examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh
