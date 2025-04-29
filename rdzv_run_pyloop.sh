pkill -9 pt_main_thread
pkill -9 python
# export PROJECT_NAME="57x288x512_node64_tp2_bs2_gc2_lr2e-5_1e-6_mwmode_fp32"
# export PROJECT_NAME="57x288x512_node64_tp2_bs2_gc2_lr4e-5_wd1e-2_final"
export PROJECT_NAME="57x288x512_node64_tp2_bs2_gc2_lr6e-5_wd1e-2_ac_fp32_bf16_new"
# export PROJECT_NAME="test_64node_master_worker"
export PROJECT_EXP_NAME="all_data"
export PROJECT_DIR="/work/share1/checkpoint/gyy/osp/$PROJECT_NAME"

bash examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh
