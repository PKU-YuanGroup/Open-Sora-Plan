export PROJECT_NAME="105x288x512_node48_tp4_bs4_gc2_lr4e-5_wd1e-2"
# export PROJECT_NAME="test_4node_final"
export PROJECT_EXP_NAME="all_data"
export PROJECT_DIR="/work/share1/checkpoint/gyy/osp/$PROJECT_NAME"

bash examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh
