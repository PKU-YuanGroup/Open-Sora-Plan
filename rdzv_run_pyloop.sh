pkill -9 pt_main_thread
pkill -9 python
ps aux | grep '[p]ython' | awk '{print $2}' | xargs -r kill -9
sync; echo 3 > /proc/sys/vm/drop_caches
sleep 10s
echo "start process..."
# export PROJECT_NAME="57x288x512_node64_tp2_bs2_gc2_lr2e-5_1e-6_mwmode_fp32"
# export PROJECT_NAME="57x288x512_node64_tp2_bs2_gc2_lr4e-5_wd1e-2_final"
export PROJECT_NAME="121x576x1024_node64_tp4_bs1_gc4_lr1e-5_wd1e-2_hq"
# export PROJECT_NAME="test_64node_master_worker"
export PROJECT_EXP_NAME="all_data"
export PROJECT_DIR="/work/share1/checkpoint/gyy/osp/$PROJECT_NAME"

bash examples/opensoraplan1.5/pretrain_opensoraplan1_5_rdzv.sh
